from numba import njit


class Compiler:

    @staticmethod
    def compile(code):
        # return code
        # Compilation is disabled for now
        return njit(fastmath=True)(code)

    def get_iterator():
        return range