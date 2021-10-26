from numba import njit


class Compiler:

    @classmethod
    def set_enabled(cls, b):
        cls.is_enabled = b

    @classmethod
    def compile(cls, code):
        if cls.is_enabled:
            print("Compiling with numba")
            return njit(fastmath=True)(code)
        return code

    def get_iterator():
        return range

Compiler.set_enabled(True)