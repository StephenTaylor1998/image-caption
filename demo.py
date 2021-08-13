#
# args = (1, 2, 3, 4)
#
# print(args)
#
# print(*args)
#
# print(1, 2, 3, 4)

def test(a=None, b=None):
    print(a, b)


kwargs = {'a': 1, 'b': 2}
# test(kwargs)

test(**kwargs)
test(a=1, b=2)
