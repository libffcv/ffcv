from numba import njit


class Compiler:

    @staticmethod
    def compile(code):
        # Compilation is disabled for now
        return njit(fastmath=True)(code)