#Type cast and utilities

#Update a dict/set and return it
def update_return(a, b):
    a.update(b)
    return a


#All-time favorite listify
def listify(x):
    if type(x) == type([]):
        return x
    else:
        return [x]
    