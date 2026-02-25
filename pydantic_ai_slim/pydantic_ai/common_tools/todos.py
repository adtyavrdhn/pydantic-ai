"""TodoToolset — structured task management tools for agents."""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from typing_extensions import Self

from ..exceptions import ModelRetry
from ..toolsets.function import FunctionToolset

__all__ = ('Todo', 'TodoStatus', 'TodoStorage', 'InMemoryTodoStorage', 'TodoToolset')

TodoStatus = Literal['pending', 'in_progress', 'completed']


@dataclass
class Todo:
    """A single todo item tracked by the agent."""

    id: int
    """Unique identifier assigned by storage."""
    content: str
    """Task description."""
    status: TodoStatus
    """Current status: pending, in_progress, or completed."""
    active_form: str
    """Present continuous form shown while in progress (e.g. "Running tests")."""
    depends_on: list[int] = field(default_factory=lambda: list[int]())
    """IDs of prerequisite todos that must complete first."""


@runtime_checkable
class TodoStorage(Protocol):
    """Protocol for todo persistence backends."""

    async def list(self) -> list[Todo]: ...
    async def get(self, todo_id: int) -> Todo | None: ...
    async def create(self, content: str, active_form: str, depends_on: list[int]) -> Todo: ...
    async def update(self, todo: Todo) -> None: ...
    async def delete(self, todo_id: int) -> None: ...


class InMemoryTodoStorage:
    """In-memory todo storage with auto-incrementing IDs."""

    def __init__(self) -> None:
        self._todos: dict[int, Todo] = {}
        self._next_id: int = 1

    async def list(self) -> list[Todo]:
        return list(self._todos.values())

    async def get(self, todo_id: int) -> Todo | None:
        return self._todos.get(todo_id)

    async def create(self, content: str, active_form: str, depends_on: list[int]) -> Todo:
        todo = Todo(
            id=self._next_id,
            content=content,
            status='pending',
            active_form=active_form,
            depends_on=depends_on,
        )
        self._todos[self._next_id] = todo
        self._next_id += 1
        return todo

    async def update(self, todo: Todo) -> None:
        self._todos[todo.id] = todo

    async def delete(self, todo_id: int) -> None:
        self._todos.pop(todo_id, None)


class TodoToolset(FunctionToolset[Any]):
    """Toolset providing structured task management tools for agents.

    Tracks todos with status, dependencies, and cycle detection.
    Each `agent.run()` gets fresh state by default; pass a custom
    `storage` to share state across runs.

    Usage:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai.common_tools.todos import TodoToolset

        agent = Agent('openai:gpt-4o', toolsets=[TodoToolset()])
        ```
    """

    def __init__(self, *, storage: TodoStorage | None = None, id: str | None = None):
        """Create a new todo toolset.

        Args:
            storage: Custom storage backend. If `None`, a fresh
                `InMemoryTodoStorage` is created for each agent run.
            id: Optional unique ID for the toolset.
        """
        super().__init__(id=id)
        self._user_storage = storage
        self._storage: TodoStorage = storage or InMemoryTodoStorage()
        self._register_tools()

    async def __aenter__(self) -> Self:
        if self._user_storage is None:
            self._storage = InMemoryTodoStorage()
        return await super().__aenter__()

    def _register_tools(self) -> None:
        self._register_list_todos()
        self._register_create_todo()
        self._register_update_todo()
        self._register_complete_todo()
        self._register_delete_todo()

    def _format_todo_list(self, todos: list[Todo]) -> str:
        """Format a list of todos for display."""
        if not todos:
            return 'No todos.'

        completed_ids = {t.id for t in todos if t.status == 'completed'}
        lines: list[str] = []
        for t in todos:
            icon = {'pending': '[ ]', 'in_progress': '[~]', 'completed': '[x]'}[t.status]
            line = f'{icon} {t.id}: {t.content}'
            if t.status == 'in_progress':
                line += f' | {t.active_form}'
            blocked_by = [d for d in t.depends_on if d not in completed_ids]
            if blocked_by:
                line += f' (blocked by: {", ".join(str(d) for d in blocked_by)})'
            lines.append(line)
        return '\n'.join(lines)

    async def _would_create_cycle(self, source_id: int, new_dep_ids: list[int]) -> int | None:
        """Return the dep ID that would create a cycle, or None."""
        todos = await self._storage.list()
        deps_map = {t.id: t.depends_on for t in todos}
        for dep_id in new_dep_ids:
            visited: set[int] = set()
            queue = collections.deque([dep_id])
            while queue:
                current = queue.popleft()
                if current == source_id:
                    return dep_id
                if current in visited:
                    continue
                visited.add(current)
                queue.extend(deps_map.get(current, []))
        return None

    def _register_list_todos(self) -> None:
        async def list_todos() -> str:
            """List all todos with their status, dependencies, and blocked state."""
            todos = await self._storage.list()
            return self._format_todo_list(todos)

        self.tool(list_todos)

    def _register_create_todo(self) -> None:
        async def create_todo(content: str, active_form: str, depends_on: list[int] | None = None) -> str:
            """Create a new todo.

            Args:
                content: Task description.
                active_form: Present continuous form (e.g. "Running tests").
                depends_on: IDs of prerequisite todos.
            """
            dep_ids = depends_on or []
            for dep_id in dep_ids:
                if await self._storage.get(dep_id) is None:
                    raise ModelRetry(f'Dependency todo {dep_id} does not exist.')
            todo = await self._storage.create(content, active_form, dep_ids)
            return f'Created todo {todo.id}: {todo.content}'

        self.tool(create_todo)

    def _register_update_todo(self) -> None:
        async def update_todo(
            todo_id: int,
            content: str | None = None,
            active_form: str | None = None,
            status: TodoStatus | None = None,
            depends_on: list[int] | None = None,
        ) -> str:
            """Update an existing todo. Only provided fields are changed.

            Args:
                todo_id: The ID of the todo to update.
                content: New task description.
                active_form: New present continuous form.
                status: New status.
                depends_on: New list of prerequisite todo IDs.
            """
            todo = await self._storage.get(todo_id)
            if todo is None:
                raise ModelRetry(f'Todo {todo_id} not found.')

            if depends_on is not None:
                for dep_id in depends_on:
                    if dep_id == todo_id:
                        raise ModelRetry(f'Todo {todo_id} cannot depend on itself.')
                    if await self._storage.get(dep_id) is None:
                        raise ModelRetry(f'Dependency todo {dep_id} does not exist.')
                if cycle_dep := await self._would_create_cycle(todo_id, depends_on):
                    raise ModelRetry(f'Adding dependency on todo {cycle_dep} would create a cycle.')
                todo.depends_on = depends_on

            if content is not None:
                todo.content = content
            if active_form is not None:
                todo.active_form = active_form
            if status is not None:
                todo.status = status

            await self._storage.update(todo)
            return f'Updated todo {todo_id}.'

        self.tool(update_todo)

    def _register_complete_todo(self) -> None:
        async def complete_todo(todo_id: int) -> str:
            """Mark a todo as completed and return the remaining task summary.

            Args:
                todo_id: The ID of the todo to complete.
            """
            todo = await self._storage.get(todo_id)
            if todo is None:
                raise ModelRetry(f'Todo {todo_id} not found.')
            todo.status = 'completed'
            await self._storage.update(todo)
            todos = await self._storage.list()
            return self._format_todo_list(todos)

        self.tool(complete_todo)

    def _register_delete_todo(self) -> None:
        async def delete_todo(todo_id: int) -> str:
            """Delete a todo and clean up dangling dependency references.

            Args:
                todo_id: The ID of the todo to delete.
            """
            todo = await self._storage.get(todo_id)
            if todo is None:
                raise ModelRetry(f'Todo {todo_id} not found.')
            await self._storage.delete(todo_id)
            # Clean dangling references
            for t in await self._storage.list():
                if todo_id in t.depends_on:
                    t.depends_on = [d for d in t.depends_on if d != todo_id]
                    await self._storage.update(t)
            return f'Deleted todo {todo_id}.'

        self.tool(delete_todo)
