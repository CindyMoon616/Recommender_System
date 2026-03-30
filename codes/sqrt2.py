import math

# f(x)=x**2-2=0
# f'(x)=2x

def Newton(x,eps):
    if x==0:
        return False
    while True:
        x_old=x
        x=x-(x**2-2)/(2*x)
        if abs(x-x_old) < eps:
            return x

def bi_search(eps):
    left,right=1,2
    while right-left >= eps:
        mid=(left+right)/2
        if mid**2 > 2:
            right=mid
        else:
            left=mid
    return mid
            


if __name__=="__main__":
    x=Newton(-5,1e-6)
    y=bi_search(1e-6)
    print(x,y)