from typing import TypeVar, Generic, Callable

T = TypeVar('T')


class Observable(Generic[T]):
    def __init__(self, content: T):
        self.data: T = content
        self.listeners = []

    def set(self, content: T):
        old_content = self.get()
        if old_content != content:
            self.data = content
            self._notify()
        else:
            self.data = content

    def get(self) -> T:
        return self.data

    def subscribe(self, listener: Callable[[T], None]):
        self.listeners.append(listener)

    def _notify(self):
        data = self.get()
        for listener in self.listeners:
            listener(data)
