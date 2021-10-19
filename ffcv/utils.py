def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]