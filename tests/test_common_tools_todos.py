"""Tests for the TodoToolset."""

from __future__ import annotations

import pytest

from pydantic_ai import Agent, ToolCallPart
from pydantic_ai._run_context import RunContext
from pydantic_ai._tool_manager import ToolManager
from pydantic_ai.common_tools.todos import InMemoryTodoStorage, TodoToolset
from pydantic_ai.exceptions import ToolRetryError
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

pytestmark = pytest.mark.anyio


def build_run_context() -> RunContext[None]:
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
    )


# --- InMemoryTodoStorage tests ---


class TestInMemoryTodoStorage:
    async def test_create_and_get(self):
        storage = InMemoryTodoStorage()
        todo = await storage.create('Write tests', 'Writing tests', [])
        assert todo.id == 1
        assert todo.content == 'Write tests'
        assert todo.status == 'pending'

        fetched = await storage.get(1)
        assert fetched is not None
        assert fetched.content == 'Write tests'

    async def test_auto_increment_ids(self):
        storage = InMemoryTodoStorage()
        t1 = await storage.create('First', 'Doing first', [])
        t2 = await storage.create('Second', 'Doing second', [])
        t3 = await storage.create('Third', 'Doing third', [])
        assert (t1.id, t2.id, t3.id) == (1, 2, 3)

    async def test_list(self):
        storage = InMemoryTodoStorage()
        await storage.create('A', 'Doing A', [])
        await storage.create('B', 'Doing B', [])
        todos = await storage.list()
        assert len(todos) == 2
        assert [t.content for t in todos] == ['A', 'B']

    async def test_update(self):
        storage = InMemoryTodoStorage()
        todo = await storage.create('Original', 'Doing original', [])
        todo.content = 'Updated'
        todo.status = 'completed'
        await storage.update(todo)
        fetched = await storage.get(1)
        assert fetched is not None
        assert fetched.content == 'Updated'
        assert fetched.status == 'completed'

    async def test_delete(self):
        storage = InMemoryTodoStorage()
        await storage.create('To delete', 'Deleting', [])
        await storage.delete(1)
        assert await storage.get(1) is None
        assert await storage.list() == []

    async def test_delete_missing_is_noop(self):
        storage = InMemoryTodoStorage()
        await storage.delete(999)  # should not raise

    async def test_get_missing_returns_none(self):
        storage = InMemoryTodoStorage()
        assert await storage.get(42) is None


# --- Tool tests ---


async def _make_tool_manager(toolset: TodoToolset) -> ToolManager[None]:
    ctx = build_run_context()
    return await ToolManager[None](toolset).for_run_step(ctx)


class TestListTodos:
    async def test_empty(self):
        toolset = TodoToolset()
        tm = await _make_tool_manager(toolset)
        result = await tm.handle_call(ToolCallPart(tool_name='list_todos', args={}))
        assert result == 'No todos.'

    async def test_with_items(self):
        storage = InMemoryTodoStorage()
        await storage.create('Setup DB', 'Setting up DB', [])
        todo2 = await storage.create('Run migrations', 'Running migrations', [])
        todo2.status = 'in_progress'
        await storage.update(todo2)
        todo3 = await storage.create('Deploy', 'Deploying', [2])
        await storage.update(todo3)

        toolset = TodoToolset(storage=storage)
        tm = await _make_tool_manager(toolset)
        result = await tm.handle_call(ToolCallPart(tool_name='list_todos', args={}))

        assert '[ ] 1: Setup DB' in result
        assert '[~] 2: Run migrations | Running migrations' in result
        assert '[ ] 3: Deploy (blocked by: 2)' in result

    async def test_completed_deps_not_shown_as_blocked(self):
        storage = InMemoryTodoStorage()
        t1 = await storage.create('Step 1', 'Doing step 1', [])
        t1.status = 'completed'
        await storage.update(t1)
        await storage.create('Step 2', 'Doing step 2', [1])

        toolset = TodoToolset(storage=storage)
        tm = await _make_tool_manager(toolset)
        result = await tm.handle_call(ToolCallPart(tool_name='list_todos', args={}))

        assert 'blocked by' not in result


class TestCreateTodo:
    async def test_basic(self):
        toolset = TodoToolset()
        tm = await _make_tool_manager(toolset)
        result = await tm.handle_call(
            ToolCallPart(tool_name='create_todo', args={'content': 'Write docs', 'active_form': 'Writing docs'})
        )
        assert result == 'Created todo 1: Write docs'

    async def test_with_valid_deps(self):
        storage = InMemoryTodoStorage()
        await storage.create('Prereq', 'Doing prereq', [])
        toolset = TodoToolset(storage=storage)
        tm = await _make_tool_manager(toolset)
        result = await tm.handle_call(
            ToolCallPart(
                tool_name='create_todo',
                args={'content': 'Follow up', 'active_form': 'Following up', 'depends_on': [1]},
            )
        )
        assert result == 'Created todo 2: Follow up'

    async def test_invalid_dep_raises_model_retry(self):
        toolset = TodoToolset()
        tm = await _make_tool_manager(toolset)
        with pytest.raises(ToolRetryError, match='does not exist'):
            await tm.handle_call(
                ToolCallPart(
                    tool_name='create_todo',
                    args={'content': 'Bad', 'active_form': 'Doing bad', 'depends_on': [999]},
                )
            )


class TestUpdateTodo:
    async def test_update_content(self):
        storage = InMemoryTodoStorage()
        await storage.create('Original', 'Doing original', [])
        toolset = TodoToolset(storage=storage)
        tm = await _make_tool_manager(toolset)
        result = await tm.handle_call(
            ToolCallPart(tool_name='update_todo', args={'todo_id': 1, 'content': 'Revised'})
        )
        assert result == 'Updated todo 1.'
        updated = await storage.get(1)
        assert updated is not None
        assert updated.content == 'Revised'

    async def test_not_found(self):
        toolset = TodoToolset()
        tm = await _make_tool_manager(toolset)
        with pytest.raises(ToolRetryError, match='not found'):
            await tm.handle_call(ToolCallPart(tool_name='update_todo', args={'todo_id': 99}))

    async def test_self_dependency(self):
        storage = InMemoryTodoStorage()
        await storage.create('Task', 'Doing task', [])
        toolset = TodoToolset(storage=storage)
        tm = await _make_tool_manager(toolset)
        with pytest.raises(ToolRetryError, match='cannot depend on itself'):
            await tm.handle_call(
                ToolCallPart(tool_name='update_todo', args={'todo_id': 1, 'depends_on': [1]})
            )

    async def test_direct_cycle(self):
        storage = InMemoryTodoStorage()
        await storage.create('A', 'Doing A', [])
        await storage.create('B', 'Doing B', [1])  # B depends on A
        toolset = TodoToolset(storage=storage)
        tm = await _make_tool_manager(toolset)
        # Try to make A depend on B — would create A -> B -> A cycle
        with pytest.raises(ToolRetryError, match='would create a cycle'):
            await tm.handle_call(
                ToolCallPart(tool_name='update_todo', args={'todo_id': 1, 'depends_on': [2]})
            )

    async def test_transitive_cycle(self):
        storage = InMemoryTodoStorage()
        await storage.create('A', 'Doing A', [])
        await storage.create('B', 'Doing B', [1])  # B -> A
        await storage.create('C', 'Doing C', [2])  # C -> B -> A
        toolset = TodoToolset(storage=storage)
        tm = await _make_tool_manager(toolset)
        # Try to make A depend on C — would create A -> C -> B -> A cycle
        with pytest.raises(ToolRetryError, match='would create a cycle'):
            await tm.handle_call(
                ToolCallPart(tool_name='update_todo', args={'todo_id': 1, 'depends_on': [3]})
            )

    async def test_invalid_dep(self):
        storage = InMemoryTodoStorage()
        await storage.create('Task', 'Doing task', [])
        toolset = TodoToolset(storage=storage)
        tm = await _make_tool_manager(toolset)
        with pytest.raises(ToolRetryError, match='does not exist'):
            await tm.handle_call(
                ToolCallPart(tool_name='update_todo', args={'todo_id': 1, 'depends_on': [999]})
            )


class TestCompleteTodo:
    async def test_complete(self):
        storage = InMemoryTodoStorage()
        await storage.create('Task A', 'Doing A', [])
        await storage.create('Task B', 'Doing B', [])
        toolset = TodoToolset(storage=storage)
        tm = await _make_tool_manager(toolset)
        result = await tm.handle_call(ToolCallPart(tool_name='complete_todo', args={'todo_id': 1}))
        # Should return the full todo list with task 1 marked as completed
        assert '[x] 1: Task A' in result
        assert '[ ] 2: Task B' in result

        completed = await storage.get(1)
        assert completed is not None
        assert completed.status == 'completed'

    async def test_not_found(self):
        toolset = TodoToolset()
        tm = await _make_tool_manager(toolset)
        with pytest.raises(ToolRetryError, match='not found'):
            await tm.handle_call(ToolCallPart(tool_name='complete_todo', args={'todo_id': 99}))


class TestDeleteTodo:
    async def test_delete(self):
        storage = InMemoryTodoStorage()
        await storage.create('To delete', 'Deleting', [])
        toolset = TodoToolset(storage=storage)
        tm = await _make_tool_manager(toolset)
        result = await tm.handle_call(ToolCallPart(tool_name='delete_todo', args={'todo_id': 1}))
        assert result == 'Deleted todo 1.'
        assert await storage.get(1) is None

    async def test_cleans_dangling_deps(self):
        storage = InMemoryTodoStorage()
        await storage.create('Prereq', 'Doing prereq', [])
        await storage.create('Dependent', 'Doing dependent', [1])
        toolset = TodoToolset(storage=storage)
        tm = await _make_tool_manager(toolset)
        await tm.handle_call(ToolCallPart(tool_name='delete_todo', args={'todo_id': 1}))
        dependent = await storage.get(2)
        assert dependent is not None
        assert dependent.depends_on == []

    async def test_not_found(self):
        toolset = TodoToolset()
        tm = await _make_tool_manager(toolset)
        with pytest.raises(ToolRetryError, match='not found'):
            await tm.handle_call(ToolCallPart(tool_name='delete_todo', args={'todo_id': 99}))


# --- Per-run isolation tests ---


class TestPerRunIsolation:
    async def test_default_storage_fresh_per_run(self):
        toolset = TodoToolset()

        # First run
        async with toolset:
            tm = await _make_tool_manager(toolset)
            await tm.handle_call(
                ToolCallPart(tool_name='create_todo', args={'content': 'Run 1 task', 'active_form': 'Working'})
            )
            result = await tm.handle_call(ToolCallPart(tool_name='list_todos', args={}))
            assert 'Run 1 task' in result

        # Second run — should be fresh
        async with toolset:
            tm = await _make_tool_manager(toolset)
            result = await tm.handle_call(ToolCallPart(tool_name='list_todos', args={}))
            assert result == 'No todos.'

    async def test_custom_storage_shared_across_runs(self):
        storage = InMemoryTodoStorage()
        toolset = TodoToolset(storage=storage)

        # First run
        async with toolset:
            tm = await _make_tool_manager(toolset)
            await tm.handle_call(
                ToolCallPart(tool_name='create_todo', args={'content': 'Shared task', 'active_form': 'Working'})
            )

        # Second run — should see the task from first run
        async with toolset:
            tm = await _make_tool_manager(toolset)
            result = await tm.handle_call(ToolCallPart(tool_name='list_todos', args={}))
            assert 'Shared task' in result


# --- Integration tests ---


class TestIntegration:
    async def test_tool_definitions(self):
        toolset = TodoToolset()
        ctx = build_run_context()
        tm = await ToolManager[None](toolset).for_run_step(ctx)
        tool_names = sorted(td.name for td in tm.tool_defs)
        assert tool_names == ['complete_todo', 'create_todo', 'delete_todo', 'list_todos', 'update_todo']

    async def test_agent_end_to_end(self):
        toolset = TodoToolset()
        agent = Agent(
            TestModel(call_tools=['list_todos'], custom_output_text='Done'),
            toolsets=[toolset],
        )

        result = await agent.run('List my tasks')
        assert result.output == 'Done'
