from numba import njit


class Compiler:

    @staticmethod
    def compile(code):
        # Compilation is disabled for now
        return code
        return njit(fastmath=True)(code)