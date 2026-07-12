import verifiers.v1 as vf


class AdditionDemoData(vf.TaskData):
    """One addition problem and its reference answer."""

    left: int
    right: int
    answer: int


class AdditionDemoTask(vf.Task[AdditionDemoData]):
    """Score one model answer."""

    @vf.reward(weight=1.0)
    async def exact_answer(self, trace: vf.Trace) -> float:
        reply = (trace.last_reply or "").strip()
        return float(reply == str(self.data.answer))


class AdditionDemoConfig(vf.TasksetConfig):
    num_tasks: int = 5
    """How many tasks to build."""


class AdditionDemoTaskset(vf.Taskset[AdditionDemoTask, AdditionDemoConfig]):
    def load(self) -> list[AdditionDemoTask]:
        return [
            AdditionDemoTask(
                AdditionDemoData(
                    idx=i,
                    prompt=(
                        "Answer with only the final integer, no words.\n\n"
                        f"What is {i + 2} + {(i + 1) * 3}?"
                    ),
                    left=i + 2,
                    right=(i + 1) * 3,
                    answer=(i + 2) + ((i + 1) * 3),
                ),
                self.config.task,
            )
            for i in range(self.config.num_tasks)
        ]
