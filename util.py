




def pareto_dominates(r11,r12, r21,r22):
    if r11 > r21 and r12 >= r22:
        return True
    elif r12 > r22 and r11 >= r21:
        return True
    else:
        return False