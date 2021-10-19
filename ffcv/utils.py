def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def is_power_of_2(n):
    return (n & (n-1) == 0) and n != 0
