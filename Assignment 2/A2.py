#Chirag Jain 2019CS10342
#Sarthak Singla 2019CS10397

import sys
import json
import copy

sys.setrecursionlimit(100002)
sys.stdout.flush()

inputFile = sys.argv[1]
lines = []

#read file 
with open(inputFile) as file:
    lines = file.read().splitlines()
    lines = [line for line in lines if line]

#set part
if(lines[0] == 'N,D,m,a,e'):
    part = 1
elif(lines[0] == 'N,D,m,a,e,S,T'):
    part = 2
else:
    print('Wrong input format')
    sys.exit()

lines=lines[1:]

N, D, m, a, e, r, S, R, hcf = 0, 0, 0, 0, 0, 0, 0, 0, 1
gsoln=[]
bsoln=[]
score = 0
prf = []


bktrk=0

#timeout handling
import signal
from contextlib import contextmanager
class TimeoutException(Exception): pass
@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


#Helps i sorting the list according to not rested and rested 
def shiftOrder(shift, rest):
    i=0
    j=len(shift)-1

    while(i<j):
        if(not rest[shift[i]]):
            i+=1
        elif(rest[shift[j]]):
            j-=1
        else:
            shift[i], shift[j] = shift[j], shift[i]

#returns the order in which variables are to be assigned for a particular day
def variableOrder(d, morn_even, after, rested, rest):
    
    order = []
    shiftOrder(morn_even, rest)
    shiftOrder(after, rest)

    
    if(d%7==6):
        i=0
        j=0
        while(i<len(morn_even) or j<len(after)):
            if(i<len(morn_even) and not rest[morn_even[i]]):
                order.append(morn_even[i])
                i+=1
            elif(j<len(after) and not rest[after[j]]):
                order.append(after[j])
                j+=1
            elif(i<len(morn_even)):
                order.append(morn_even[i])
                i+=1
            else:
                order.append(after[j])
                j+=1
    else:
        order+=morn_even
        order+=after

    order+=rested
    # print(order)
    # print(morn_even)
    # print(after)


    return order

#returns valid domain values possible for a nurse on particular day in current configuration
def domain(nurse, day, rest):
    
    global N, D, m, a, e, r, gsoln
    
    if(day%7==6 and not rest[nurse]): # last day
        if(r):
            return [4]
        return []


    dom = []


    if part==1:        
        if((rest[nurse] or r==1) and a!=1):
            if(a):
                dom.append(2)
            if(r):
                dom.append(4)
        else:
            if(r):
                dom.append(4)
            if(a):
                dom.append(2)

        if(gsoln[day-1][nurse] in [1,3]): # m, e
            if(e):
                dom.append(3)
            return dom
        
        else: # a, r
            if(m==1 and e):
                dom.append(3)
                dom.append(1)
            else:
                if(m):
                    dom.append(1)
                if(e):
                    dom.append(3)
        return dom


    if(r):
        dom.append(4)
    if(a):
        dom.append(2)
    if(e):
        dom.append(3)
    if(m):
        if(gsoln[day-1][nurse] not in [1,3]):
            dom.append(1)
    
    
    
    final_dom = []
    
    
    # if(gsoln[0][nurse] in [1,3]):
    if(nurse<S):
        # if(day%7==5 and day+2==D or day+4==D):
        #     if(4 in dom):
        #         final_dom.append(4)
        #     if(1 in dom):
        #         final_dom.append(1)
        #     if(3 in dom):
        #         final_dom.append(3)
        #     if(2 in dom):
        #         final_dom.append(2)
        # else:
        if(3 in dom):
            final_dom.append(3)
        if(1 in dom):
            final_dom.append(1)
        if(4 in dom):
            final_dom.append(4)
        if(2 in dom):
            final_dom.append(2)
    
    else:
        if((rest[nurse] or r==1) and a!=1):
            if(2 in dom):
                final_dom.append(2)
            if(4 in dom):
                final_dom.append(4)
        else:
            if(4 in dom):
                final_dom.append(4)
            if(2 in dom):
                final_dom.append(2)
        
        if(1 in dom and 3 in dom):
            if(m==1 and e):
                final_dom.append(3)
                final_dom.append(1)
            else:
                if(m):
                    final_dom.append(1)
                if(e):
                    final_dom.append(3)
            # if(e==1 and m):
            #     final_dom.append(1)
            #     final_dom.append(3)
            # else:
            #     if(e):
            #         final_dom.append(3)
            #     if(m):
            #         final_dom.append(1)
        else:
            if(3 in dom):
                final_dom.append(3)
            if(1 in dom):
                final_dom.append(1)

    return final_dom

#convert our 2D array to a dictionary according to our preference order
def convertsol():
    global bsoln, D, N, prf
    output = {}

    for day in range(D):
        for i in range(N):
            nurse = prf[i][1]
            if(bsoln[day][nurse]==1):
                for j in range(hcf):
                    output['N'+str(i*hcf+j)+'_'+str(day)] = 'M'
            elif(bsoln[day][nurse]==2):
                for j in range(hcf):
                    output['N'+str(i*hcf+j)+'_'+str(day)] = 'A'
            elif(bsoln[day][nurse]==3):
                for j in range(hcf):
                    output['N'+str(i*hcf+j)+'_'+str(day)] = 'E'
            else:
                for j in range(hcf):
                    output['N'+str(i*hcf+j)+'_'+str(day)] = 'R'
    
    return output

#dump our solution to "solution.json" file
def dumpsolution():
    soln_list = convertsol()
    # print(bsoln)
    with open("solution.json" , 'w') as file:
        json.dump(soln_list,file)
        file.write("\n")
    if(part==1):
        sys.exit()

#check if this soln better than current soln and update accodingly
def bestsol():
    global bsoln, gsoln, score, prf, bktrk
    bktrk+=1
    preff = []
    tempScore = 0
    for nurse in range(N):
        preff.append([0,nurse])
        for day in range(D):
            if(gsoln[day][nurse]%2): 
                preff[nurse][0]+=1
    
    preff.sort(reverse=True)

    for i in range(S):
        tempScore+=preff[i][0]
    
    if(tempScore>score or bsoln==[]):
        bsoln=copy.deepcopy(gsoln)
        prf = preff[:]
        score = tempScore
        # print(bktrk,score*hcf)
        dumpsolution()


def gcd(a,b):
    if(a==0 and b==0): return 1
    if(b==0): return a
    return gcd(b,a%b)

#main solver which calls our functions to get answer
def solver(nor, day, rest, order, RN):
    global N, D, m, a, e, r, gsoln
    
    if(nor==0):
        morn_even = []
        after = []
        rested = []
        m = 0
        a = 0
        e = 0
        r = 0
        # if(part==2):
        #     for i in range(S):
        #         if(gsoln[day-1][i]==1):
        #             morn_even.append(i)
        #             m+=1
        #         elif(gsoln[day-1][i]==2):
        #             after.append(i)
        #             a+=1
        #         elif(gsoln[day-1][i]==3):
        #             morn_even.append(i)
        #             e+=1
        #         elif(gsoln[day-1][i]==4):
        #             rested.append(i)
        #             r+=1
        #     order1 = variableOrder(day, morn_even, after, rested, rest)
        #     morn_even = []
        #     after = []
        #     rested = []
        #     for i in range(S,N):
        #         if(gsoln[day-1][i]==1):
        #             morn_even.append(i)
        #             m+=1
        #         elif(gsoln[day-1][i]==2):
        #             after.append(i)
        #             a+=1
        #         elif(gsoln[day-1][i]==3):
        #             morn_even.append(i)
        #             e+=1
        #         elif(gsoln[day-1][i]==4):
        #             rested.append(i)
        #             r+=1
        #     order2 = variableOrder(day, morn_even, after, rested, rest)
        #     order1 = order1+order2

        
        if(part):
            for i in range(N):
                if(gsoln[day-1][i]==1):
                    morn_even.append(i)
                    m+=1
                elif(gsoln[day-1][i]==2):
                    after.append(i)
                    a+=1
                elif(gsoln[day-1][i]==3):
                    morn_even.append(i)
                    e+=1
                elif(gsoln[day-1][i]==4):
                    rested.append(i)
                    r+=1
            order1 = variableOrder(day, morn_even, after, rested, rest)
            
        # print(m,a,e,r, gsoln[day-1], day)
        # order1 = variableOrder(day, morn_even, after, rested, rest)
        if(day%7==0): # if day 0 and nurse 0 reset RN
            RN=N
        
    else:
        order1 = order
    
    if(day+(6-(day%7))<D): # less slots available than reqd
        if(RN>R*(6-(day%7))+r):
            # print(nor, day, RN, gsoln[day-1], gsoln[day], rest)
            return None
    

    
    # print(order1[nor], day, gsoln[day], m,a,e,r)
    
    nurse = order1[nor] # actual nurse number
    # print(nor, nurse, day)
    
    values = domain(nurse, day, rest)
    # if(values==[]):
    #     print('Prob')
    #     print(nurse, day, RN, gsoln[day], gsoln[day-1],order1)
    # print(nor,nurse,day,values)

    if(nor+1==N and day%7==6):
        rest1 = [0 for i in range(N)]
    else:
        rest1 = rest

    restStatus = rest1[nurse]
    for value in values:
        gsoln[day][nurse] = value
        if(value==1):
            m-=1
        elif(value==2):
            a-=1
        elif(value==3):
            e-=1
        elif(value==4):
            r-=1
            if(not (nor+1==N and day%7==6)):
                rest1[nurse]=1
        # temp = True
        if(nor+1==N):
            if(day+1==D):
                # print(m,a,e,r,gsoln[day])
                bestsol()
                # print(bktrk,m,a,e,r)
            else:
                if(restStatus!=rest1[nurse]):
                    solver(0, day+1, rest1, order1, RN-1)
                else: 
                    solver(0, day+1, rest1, order1,RN) 
                m,a,e,r=0,0,0,0
        else:
            day1 = day
            if(restStatus!=rest1[nurse]):
                solver(nor+1, day1, rest1, order1, RN-1)
            else:
                solver(nor+1, day1, rest1, order1, RN)
        # if(temp):
        #     return True
        
        if(value==1):
            m+=1
        elif(value==2):
            a+=1
        elif(value==3):
            e+=1
        elif(value==4):
            r+=1
            rest1[nurse] = restStatus
        gsoln[day][nurse] = 0

    gsoln[day][nurse] = 0
    
    return None

#Initialize values for part2
#and set values of nurses for day0
def part2initialize (line):
    global N, D, m, a, e, S, r, R, gsoln, bsoln, score,prf, hcf
    N, D, m, a, e, S, T = map(int, line.split(','))
    r = N - m - a - e
    R = r

    hcf = gcd(gcd(N,m), gcd(gcd(a,e), gcd(r,S)))

    N,m,a,e,r,S,R = [int(x/hcf) for x in [N,m,a,e,r,S,R]]
    
    if(r<0 or (r + a < m and D>1 ) or (7 * r < N and D >= 7)):
        return False

    # bsoln=[]
    # score=0
    # prf=[]
    
    # m_f,a_f,e_f = m,a,e
    # for x in range(m_f+e_f,-1,-1):
    #     for y in range(min(S-x,a_f)+1):
    #         print("hi",x,y)
    #         if(x+y<=S and x+y+r>=S):
    #             gsoln = [[0 for i in range(N)] for j in range(D)]
    #             rest = [0 for i in range(N)]
    #             mo,af,ev=m_f,a_f,e_f

    #             for i in range(S):
    #                 if(i < min(m_f,x)):
    #                     gsoln[0][i] = 1
    #                     mo-=1
    #                 elif(i < x):
    #                     gsoln[0][i] = 3
    #                     ev-=1
    #                 elif(i < x+y):
    #                     gsoln[0][i] = 2
    #                     af-=1
    #                 else:
    #                     gsoln[0][i] = 4
    #                     rest[i] = 1
                
    #             for i in range(S,N):
    #                 if(mo):
    #                     gsoln[0][i] = 1
    #                     mo-=1
    #                 elif(ev):
    #                     gsoln[0][i] = 3
    #                     ev-=1
    #                 elif(af):
    #                     gsoln[0][i] = 2
    #                     af-=1
    #                 else:
    #                     gsoln[0][i] = 4
    #                     rest[i] = 1
                
    #             if(D>1):
    #                 try:
    #                     with time_limit(10):
    #                         solver(0, 1, rest, [], N-R)
    #                 except TimeoutException as ex:
    #                     pass
    #             else:
    #                 bestsol()




    gsoln = [[0 for i in range(N)] for j in range(D)]
    rest = [0 for i in range(N)]
    bsoln=[]
    score=-1
    prf=[]
    
    morn = []
    after = []
    even = []

    for i in range(N):
        if(i < m):
            gsoln[0][i] = 1
            even.append(i)
        elif(i < e + m):
            gsoln[0][i] = 3
            after.append(i)
        elif(i < m + a + e):
            gsoln[0][i] = 2
            morn.append(i)
        else:
            gsoln[0][i] = 4
            rest[i] = 1
    
    if(D>1):
        try:
            with time_limit(max(1,T-1)):
                solver(0, 1, rest, [], N-R)
        except TimeoutException as e:
            pass
    else:
        bestsol()

#Initialize values for part1
#and set values of nurses for day0
def part1initialize (line):
    global N, D, m, a, e, S, r, R, gsoln, bsoln, score,prf
    N, D, m, a, e = map(int, line.split(','))
    r = N - m - a - e
    R = r
    S = 0

    if(r<0 or (r + a < m and D>1) or (7 * r < N and D >= 7)):
        return False
    
    gsoln = [[0 for i in range(N)] for j in range(D)]
    rest = [0 for i in range(N)]
    bsoln=[]
    score=0
    prf=[]

    morn = []
    after = []
    even = []

    for i in range(N):
        if(i < m):
            gsoln[0][i] = 1
            morn.append(i)
        elif(i < m + e):
            gsoln[0][i] = 3
            after.append(i)
        elif(i < m + a + e):
            gsoln[0][i] = 2
            even.append(i)
        else:
            gsoln[0][i] = 4
            rest[i] = 1
    
    if(D>1):
        solver(0, 1, rest, [], N-R)
    else:
        bestsol()
        

soln_list = {}
with open("solution.json" , 'w') as file:
    json.dump(soln_list,file)
    file.write("\n")

if(part==1):
    for line in lines:
        part1initialize(line)
        if bsoln==[]:#No Solution
            print('NO SOLUTION')
        else:
            dumpsolution()
elif(part==2):
    for line in lines:
        part2initialize(line)
        if(bsoln==[]):#No Solution
            print('NO SOLUTION')
        else:
            dumpsolution()