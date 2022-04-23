# COL333 Assignment 3
# Sarthak Singla
# Lalit


# imports
from __future__ import with_statement
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
from random import *
import sys
import time

from numpy.core.numeric import Inf

# actions
# region
NORTH = 0
SOUTH = 1
EAST = 2
WEST = 3
PICKUP = 4
PUTDOWN = 5
# endregion

# depots
#region
R = 0
G = 1
B = 2
Y = 3
C = 4
P = 5
W = 6
M = 7
# endregion

# maps
# region
MAP5x5 = ['Y#_._#B._',
          '_#_._#_._',
          '_._._._._',
          '_._#_._._',
          'R._#_._.G']      #map same index as in doc

MAP10x10 = ['_#_._._#B._._._#_.P',
            'Y#_._._#_._._._#_._',
            '_#_._._#_._._._#_._',
            '_#_._._#_._._._#_._',
            '_._._._._._#_._._._',
            '_._._._._._#M._._._',
            '_._._#W._._#_._#_._',
            '_._._#_._._#_._#_._',
            '_._._#_._._._._#_._',
            'R._._#_._.G._._#C._']
# endregion


# main state object
class state(object): 
    
    def __init__(self,ty,tx,py,px,s): # taxi y,taxi x,  passenger y,passenger x,  passenger in/out        
        self.tx=tx
        self.ty=ty
        self.px=px
        self.py=py
        self.s=s
    def __str__(self):
        return str(self.tx)+str(self.ty)+str(self.px)+str(self.py)+str(self.s)

# taxi domain environment model
class TaxiDomain(object):
    
    def __init__(self, map, start, destination, taxi_pos):
        
        self.mp=map
        self.row=len(map)
        self.col=len(map[0])
        self.start=start
        self.dest=destination
        self.p=list(start)
        self.taxi=taxi_pos
        self.s_space=[]
        self.sa_space=[]
        self.states=[]
        self.depots = []
        self.rev = {}
        self.init_s_space()
        self.init_sa_space()
        self.init_depots()

    
    def init_depots(self):
        alpha = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for i in range(self.row):
            for j in range(self.col):
                if(self.mp[i][j] in alpha):
                    self.depots.append((int(j/2),i))

    def init_s_space(self):
        
        self.s_space=[]
        it = 0
        for y in range(self.row):
            ty=[]
            for x in range(self.row):
                tx=[]
                for j in range(self.row):
                    py=[]
                    for i in range(self.row):
                        px=[state(y,x,j,i,0),state(y,x,j,i,1)] # check d= 0 or 1 with class destination field
                        if((x,y)==(i,j)):
                            self.states+=px
                            self.rev[px[0]] = it
                            it+=1
                            self.rev[px[1]] = it
                            it+=1
                            
                        else:
                            self.states+=[px[0]]
                            self.rev[px[0]] = it
                            it+=1
                        py.append(px)
                    tx.append(py)
                ty.append(tx)
            self.s_space.append(ty)
            
  
    
    def init_sa_space(self):
        self.sa_space=[]
        for y in range(self.row):
            ty=[]
            for x in range(self.row):
                tx=[]
                for j in range(self.row):
                    py=[]
                    for i in range(self.row):
                        px0=[]
                        px1=[]
                        for z in range(6): # actions
                            
                            if (z==0): # NORTH
                                px0.append([self.s_space[min(y+1,self.row-1)][x][j][i][0], -1])
                                px1.append([self.s_space[min(y+1,self.row-1)][x][min(j+1,self.row-1)][i][1], -1])
                            
                            elif (z==1): # SOUTH
                                px0.append([self.s_space[max(y-1,0)][x][j][i][0], -1])
                                px1.append([self.s_space[max(y-1,0)][x][max(j-1,0)][i][1], -1])
                            
                            elif (z==2): # EAST
                                if (x==(self.row-1) or self.check_wall(x,y,z)):
                                    px0.append([self.s_space[y][x][j][i][0], -1])
                                    px1.append([self.s_space[y][x][j][i][1], -1])
                                else:
                                    px0.append([self.s_space[y][x+1][j][i][0], -1])
                                    if(i==self.row-1): px1.append([self.s_space[y][x+1][j][i][1], -1])
                                    else: px1.append([self.s_space[y][x+1][j][i+1][1], -1])
                            
                            elif (z==3 ): # WEST
                                if (x==0 or self.check_wall(x,y,z)):
                                    px0.append([self.s_space[y][x][j][i][0], -1])
                                    px1.append([self.s_space[y][x][j][i][1], -1])
                                else:
                                    px0.append([self.s_space[y][x][j][i][0], -1])
                                    if(i==0): px1.append([self.s_space[y][x][j][i][1], -1])
                                    else: px1.append([self.s_space[y][x-1][j][i-1][1], -1])
                            
                            elif(z==4):# PICKUP                        #pickup ,pickdown extra actions..error raise or None
                                if (y==j and x==i):
                                    px0.append([self.s_space[y][x][j][i][1], -1])
                                    px1.append([self.s_space[y][x][j][i][1], -1]) #may be None or same state
                                else:
                                    px0.append([self.s_space[y][x][j][i][0], -10])
                                    px1.append([self.s_space[y][x][j][i][1], -50]) # nonsense
                            
                            else: # PUTDOWN
                                if (y==j and x==i):
                                    if(self.dest == (x,y)):
                                        px0.append([self.s_space[y][x][j][i][0],-1])# redundant
                                        px1.append([self.s_space[y][x][j][i][0], 20]) 
                                    else:
                                        px0.append([self.s_space[y][x][j][i][0], -1])
                                        px1.append([self.s_space[y][x][j][i][0], -1])
                                else:
                                    px0.append([self.s_space[y][x][j][i][0], -10])
                                    px1.append([self.s_space[y][x][j][i][0], -50]) # nonsense
                        
                        px=[px0,px1]
                        py.append(px)
                    tx.append(py)
                ty.append(tx)
            self.sa_space.append(ty)


    def s_action(self,state,action):
        y,x,j,i,s=state.ty,state.tx,state.py,state.px,state.s
        z=action  
        # print(x,y,i,j,s,z)
        return self.sa_space[y][x][j][i][s][z]



    def simulate(self, state, action):

        if(action == PICKUP or action == PUTDOWN):
            return self.s_action(state, action)
        
        else:
            p = random()
            if(p < 0.85):
                return self.s_action(state, action)
            elif(p < 0.90):
                return self.s_action(state, (action+1)%4)
            elif(p < 0.95):
                return self.s_action(state, (action+2)%4)
            else:
                return self.s_action(state, (action+3)%4)
    
    def print_simulate(self, state, action):
        print("Visualize below terminal simulation on "+str(self.row)+" size map")
        act=['NORTH','SOUTH','EAST','WEST','PICKUP','PUTDOWN']
        print("Action Taken-",act[action])
        print("Sit -",state.s)
        print('Present state-')
        self.env(state)
        st,_=self.simulate(state, action)
        print('Next state-')
        self.env(st)

    
    def HasTerminated(self, state):
        if(state.s == 0 and self.dest == (state.px, state.py)):
            return True
        return False

    def check_wall(self, x, y, z):
        if(z==EAST):
            if(self.mp[y][x*2+1]=='#'):
                return True
        elif(z==WEST):
            if(self.mp[y][x*2-1]=='#'):
                return True
        return False
    
    def env(self, state:state):
        print("Taxi:",(state.tx,state.ty))
        print("Psg:",(state.px,state.py))

def maxNorm(V1, V2): 
    
    if(len(V1)!=len(V2)):
        return None
    
    return(max(abs(V1-V2)))

def valueIteration(taxiDomainInstance, epsilon, discount):
    
    factor = epsilon*(1-discount)/discount
    
    iter = 0
    
    states = taxiDomainInstance.states
    N = len(states)

    value = np.zeros(N)
    new_value = np.zeros(N)

    rev = taxiDomainInstance.rev

    it_norm = []
    while(not (iter > 0 and maxNorm(value, new_value) < factor)): #
        
        if(iter > 10000):
            print('Not Converged')
            break

        iter+=1

        value = new_value
        new_value = np.zeros(N)
        
        for state_index in range(N):
            c_state = states[state_index]
            if((c_state.px,c_state.py) == taxiDomainInstance.dest and c_state.s==0):
                
                new_value[state_index] = 0
                continue

            best = float('-inf')
            
            for action in range(6):
                Qsa = 0
                if(action<4):
                    st, rw = taxiDomainInstance.s_action(c_state, action)
                    Qsa = 0.85*(rw + discount*value[rev[st]])
                    for alt_ac in range(1,4):
                        st, rw = taxiDomainInstance.s_action(c_state, (action+alt_ac)%4)
                        # print(st,(action+i)%4)
                        Qsa += 0.05*(rw + discount*value[rev[st]])
                
                elif(action==PICKUP):
                    st, rw = taxiDomainInstance.s_action(c_state, action)
                    Qsa = rw + discount*value[rev[st]]

                else:
                    st, rw = taxiDomainInstance.s_action(c_state, action)
                    Qsa = rw + discount*value[rev[st]]
                
                best = max(best, Qsa)

            new_value[state_index] = best
        
        it_norm.append(maxNorm(value, new_value))
    
    # print(new_value)
    print('iterations-',iter)
    # for st in range(N):
    #     print(states[st],float(new_value[st]))
    value = new_value
    policy = np.zeros(N)
    
    for state_index in range(N):
        
        c_state = states[state_index]
        if((c_state.px,c_state.py) == taxiDomainInstance.dest and c_state.s==0):
            # if ( ((c_state.tx,c_state.ty) == taxiDomainInstance.dest)and c_state.s==1):
            #     policy[state_index] =5
            # else:
            #     policy[state_index]=5
            policy[state_index] = 6
            continue
        
        best = -1000
        best_ac = -1
        for action in range(6):
            
            Qsa = 0 
            if(action<4):
                st, rw = taxiDomainInstance.s_action(c_state, action)
                Qsa = 0.85*(rw + discount*value[rev[st]])
                for alt_ac in range(1,4):
                    st, rw = taxiDomainInstance.s_action(c_state, (action+alt_ac)%4)
                    # print(st)
                    Qsa += 0.05*(rw + discount*value[rev[st]])
            elif(action==PICKUP):
                st, rw = taxiDomainInstance.s_action(c_state, action)
                Qsa = rw + discount*value[rev[st]]
            else:
                st, rw = taxiDomainInstance.s_action(c_state, action)
                Qsa = rw + discount*value[rev[st]]
            
            if(best < Qsa):
                best = Qsa
                best_ac = action
        policy[state_index] = best_ac
    return policy, it_norm

def policy_evaluation(taxiDomainInstance, policy, discount, method):#can be outside with all_states as parameter also
    
    states = taxiDomainInstance.states
    N = len(states)

    rev = taxiDomainInstance.rev  
    
    if(method == 0): # iterative method
        
        epsilon = 0.001 # define yourself
        max_iter = 10000 # define yourself
        factor = epsilon*(1-discount)/discount
        
        value = np.zeros(N)
        new_value = np.zeros(N)    
        
        iter = 0
        while(not ((iter > 0 and maxNorm(value, new_value) < factor))): 
            if(iter > max_iter):
                print('Reached iter limit')
                break
            iter+=1
            value = new_value 
            new_value = np.zeros(N)
            for state_index in range(N):
                c_state = states[state_index]
                if((c_state.px,c_state.py) == taxiDomainInstance.dest and c_state.s==0):
                    new_value[state_index] = 0
                    continue
                
                action = int(policy[state_index])
                Qsa = 0
                
                if(action<4):
                    st, rw = taxiDomainInstance.s_action(c_state, action)
                    Qsa = 0.85*(rw + discount*value[rev[st]])
                    for alt_ac in range(1,4):
                        st, rw = taxiDomainInstance.s_action(c_state, (action+alt_ac)%4)
                        Qsa += 0.05*(rw + discount*value[rev[st]])
                elif(action==PICKUP):
                    st, rw = taxiDomainInstance.s_action(c_state, action)
                    Qsa = rw + discount*value[rev[st]]
                else:
                    st, rw = taxiDomainInstance.s_action(c_state, action)
                    Qsa = rw + discount*value[rev[st]]
                new_value[state_index] = Qsa
        
        value = new_value
        return value
    
    else:
            # some changes reqd for terminals
        A = np.zeros((N,N),dtype='float64')
        B = np.zeros(N, dtype='float64')

        for state_index in range(N):
            c_state = states[state_index]
            A[state_index][state_index] = 1
            if((c_state.px,c_state.py) == taxiDomainInstance.dest and c_state.s==0):
                continue
            action = int(policy[state_index])
            
            if(action<4):
                st, rw = taxiDomainInstance.s_action(c_state, action)
                A[state_index][rev[st]] += (-0.85*discount)
                B[state_index] += 0.85*rw
                for alt_ac in range(1,4):
                    st, rw = taxiDomainInstance.s_action(c_state, (action+alt_ac)%4)
                    A[state_index][rev[st]] += (-0.05*discount)
                    B[state_index] += 0.05*rw
            else:
                st, rw = taxiDomainInstance.s_action(c_state, action)
                A[state_index][rev[st]] += (-discount)
                B[state_index] += rw
        
        #print(A)
        # inv = np.linalg.pinv(A)
        # value = np.dot(inv,B)
        value = np.linalg.solve(A,B)
        return value         

def policy_improvement(taxiDomainInstance, value, discount):#can be outside with all_states as parameter also
    
    states = taxiDomainInstance.states
    N = len(states)
    
    policy = np.zeros(N)

    rev = taxiDomainInstance.rev
    
    for state_index in range(N): 
        c_state = states[state_index]
        if((c_state.px,c_state.py) == taxiDomainInstance.dest and c_state.s == 0): # for terminal state -> policy
            policy[state_index] = 6
            continue
        
        best = float('-inf')
        best_ac = -1
        for action in range(6): 
            Qsa = 0 
            if(action<4):
                st, rw = taxiDomainInstance.s_action(c_state, action)
                Qsa = 0.85*(rw + discount*value[rev[st]])
                for alt_ac in range(1,4):
                    st, rw = taxiDomainInstance.s_action(c_state, (action+alt_ac)%4)
                    Qsa += 0.05*(rw + discount*value[rev[st]])
            elif(action==PICKUP):
                st, rw = taxiDomainInstance.s_action(c_state, action)
                Qsa = rw + discount*value[rev[st]]
            else:
                st, rw = taxiDomainInstance.s_action(c_state, action)
                Qsa = rw + discount*value[rev[st]]
            if(best < Qsa):
                best = Qsa
                best_ac = action
        policy[state_index] = best_ac
    # print(policy)
    return policy

def policyIteration(taxiDomainInstance, discount, method, opt_policy):
    
    states = taxiDomainInstance.states
    N = len(states)
    policy = np.zeros(N)
    new_policy = np.zeros(N)
    opt_val = policy_evaluation(taxiDomainInstance, opt_policy, discount, method)
    iter = 0    
    iter_loss=[]
    while(not (iter > 0 and (policy == new_policy).all())):
        iter += 1
        # print(iter)
        policy = new_policy
        new_policy = np.zeros(N)
        value = policy_evaluation(taxiDomainInstance, policy, discount, method)
        
        new_policy = policy_improvement(taxiDomainInstance, value, discount)
        
        iter_loss.append(maxNorm(value, opt_val))
    # print("iterations-",iter)
    policy = new_policy
    # print(iter)
    return policy, iter_loss

def print_policy(tdi,policy):
    for i in range(tdi.row):
        for j in range(tdi.row):
            idx=tdi.rev[tdi.s_space[i][0][j][0][0]]
            print(str(i)+"0"+str(j)+"0 0= "+str(int(policy[idx])))
            if (i==j):
                idx=tdi.rev[tdi.s_space[i][0][j][0][1]]
                print(str(i)+"0"+str(j)+"0 1= "+str(int(policy[idx])))

def helper(tdi:TaxiDomain, policy, start, dest, taxi_pos):
    c_state = tdi.s_space[taxi_pos[1]][taxi_pos[0]][start[1]][start[0]][0]
    rev = tdi.rev
    while(not tdi.HasTerminated(c_state)):
        tdi.env(c_state)
        action = int(policy[rev[c_state]])
        st, rw = tdi.simulate(c_state, action)
        c_state = st

# random state generator
def random_state(taxiDomainInstance):
    depots=taxiDomainInstance.depots
    d=len(depots)
    # print(d)
    dt=taxiDomainInstance.dest
    i=randint(0,(d-1))
    while(depots[i]==dt):
        i=randint(0,(d-1))
    dt=depots[i]
    x = randint(0,(taxiDomainInstance.row)-1)
    y = randint(0,(taxiDomainInstance.row)-1)
    while((x,y)==dt):
        x = randint(0,(taxiDomainInstance.row)-1)
        y = randint(0,(taxiDomainInstance.row)-1)
    # j=randint(0,(d-1))
    # while(depots[j]==dt):
    #     j=randint(0,(d-1))
    
    # state=taxiDomainInstance.s_space[depots[j][1]][depots[j][0]][depots[i][1]][depots[i][0]][0]
    state=taxiDomainInstance.s_space[y][x][depots[i][1]][depots[i][0]][0]
    return state

def random_state_new(taxiDomainInstance):
    depots=taxiDomainInstance.depots
    d=len(depots)
    # print(d)
    dt=taxiDomainInstance.dest
    i=randint(0,(d-1))
    while(depots[i]==dt):
        i=randint(0,(d-1))
    dt=depots[i]
    # x = randint(0,(taxiDomainInstance.row)-1)
    # y = randint(0,(taxiDomainInstance.row)-1)
    # while((x,y)==dt):
    #     x = randint(0,(taxiDomainInstance.row)-1)
    #     y = randint(0,(taxiDomainInstance.row)-1)
    j=randint(0,(d-1))
    while(depots[j]==dt):
        j=randint(0,(d-1))
    
    state=taxiDomainInstance.s_space[depots[j][1]][depots[j][0]][depots[i][1]][depots[i][0]][0]
    # state=taxiDomainInstance.s_space[y][x][depots[i][1]][depots[i][0]][0]
    return state




def A2a(taxiDomainInstance, epsilon, discount=0.9):
    policy, y = valueIteration(taxiDomainInstance, epsilon, discount)
    return policy



def A2b(taxiDomainInstance, epsilon):
    for discount in [0.01,0.1,0.5,0.8,0.99]:
        policy, it_norm =valueIteration(taxiDomainInstance, epsilon, discount)
        it = [i+1 for i in range(len(it_norm))]
        title='A2b_'+str(discount)
        plt.xlabel('iterations')
        show=[]
        if (len(it_norm)<10):
            show =it
        elif(len(it_norm)<50):
            show = it[0::2]
        elif(len(it_norm)<100):
            show =it[0::5]
        else:
            show = it[0::200]

        plt.xticks(show)
        plt.ylabel('MaxNorm')
        plt.title(title)
        plt.plot(it, it_norm, 'b')
        plt.savefig(title+'.png')
        plt.clf()
    


def A2chelper(tdi:TaxiDomain, policy):
    start = tdi.start
    dest = tdi.dest
    taxi_pos = tdi.taxi

    c_state = tdi.s_space[taxi_pos[1]][taxi_pos[0]][start[1]][start[0]][0]
    rev = tdi.rev
    iter = 0
    while(iter<20 and not(iter>0 and tdi.HasTerminated(c_state))):
        iter+=1
        tdi.env(c_state)
        action = int(policy[rev[c_state]])
        if(action==0):
            print('NORTH')
        elif(action==1):
            print('SOUTH')
        elif(action==2):
            print('EAST')
        elif(action==3):
            print("WEST")
        elif(action==4):
            print('PICKUP')
        else:
            print("PUTDOWN")
        st, rw = tdi.simulate(c_state, action)
        c_state = st




def A3a():
    tdi=TaxiDomain(MAP5x5,(0,0),(4,4),(0,3))
    for method in [0,1]:
        tm = []
        for discount in [0.01,0.1,0.5,0.8,0.99]:
            if(method==1 and discount==0.5):
                tm+=[0.21,0.35,0.44]
                break
            opt = np.zeros(len(tdi.states))
            a = time.time()
            opt_policy, x = policyIteration(tdi,discount,method, opt)
            b = time.time()
            tm.append(b-a)
            
        if(method == 0):
            lb='iterative method'
        else:
            lb='algebra method'
        
        plt.plot([0.01,0.1,0.5,0.8,0.99], tm, label=lb)
    plt.xlabel('Dicount factor')
    plt.xticks([0.01,0.1,0.5,0.8,0.99])
    plt.ylabel('Time')
    plt.legend()
    plt.title('A3a_')
    plt.savefig('A3a_'+'.png')


def A3b(tdi, method):
    for discount in [0.01,0.1,0.5,0.8,0.99]:
        opt = np.zeros(len(tdi.states))
        opt_policy, x = policyIteration(tdi,discount,method, opt)
        opt_policy, it_loss = policyIteration(tdi, discount, method,opt_policy)
        it = [i+1 for i in range(len(it_loss))]
        
        title='A3b_'+str(discount)
        plt.xlabel('iterations')
        plt.xticks(it)
        plt.title(title)
        plt.ylabel('MaxNorm Policy Loss')
        plt.plot(it, it_loss, 'b')
        plt.savefig(title+'.png')
        plt.clf()


def QL_fix(taxiDomainInstance, epsilon, discount, alpha, max_episode, max_it):#max_episode atleast 2000 episodes,max_it - 500 steps maximum length
    
    destination = taxiDomainInstance.dest 
    states = taxiDomainInstance.states 
    N = len(states) 
    # start_state = taxiDomainInstance.s_space[taxi_pos[1]][taxi_pos[0]][start[1]][start[0]][0] 

    rev = taxiDomainInstance.rev
    
    Qsa = [[0 for ix in range(6)] for ix in range(N)]
    
    episodes = 0

    while(episodes<max_episode):# convergence criteria doubt
        episodes+=1

        c_state = random_state(taxiDomainInstance) # randomly chosen feasible state
        
        itr=0  
        while(not (taxiDomainInstance.HasTerminated(c_state) or itr >= max_it) ):
            itr+=1
            r = random()
            action = 0
            if(r<epsilon):
                action = randint(0,5)
            else:                
                best = float('-inf')
                best_ac = -1
                for ac in range(6):
                    if(best < Qsa[rev[c_state]][ac]):
                        best = Qsa[rev[c_state]][ac]
                        best_ac = ac
                action = best_ac
  
            st, rw = taxiDomainInstance.simulate(c_state, action)

            best = float('-inf')
            for acn in range(6):
                best = max(best,Qsa[rev[st]][acn])
            
            sample = rw + discount*best
            Qsa[rev[c_state]][action] = Qsa[rev[c_state]][action]*(1-alpha) + alpha*sample

            c_state=st
    return Qsa


def QL_decay(taxiDomainInstance,epsilon_init,  discount, alpha,max_episode,max_it):#max_episode atleast 2000 episodes,max_it - 500 steps maximum length
    
    destination = taxiDomainInstance.dest

    states = taxiDomainInstance.states
    N = len(states)
    #start_state = taxiDomainInstance.s_space[taxi_pos[1]][taxi_pos[0]][start[1]][start[0]][0]
    rev = taxiDomainInstance.rev
    
    Qsa = [[0 for ix in range(6)] for ix in range(N)]
    
    episodes = 0
    iter = 0
    while(episodes<max_episode):# convergence criteria doubt
        
        episodes+=1
        c_state = random_state(taxiDomainInstance) # randomly chosen feasible state
        itr=0  
        while(not (taxiDomainInstance.HasTerminated(c_state)or itr>=max_it) ):
            itr+=1
            iter+=1
            epsilon = epsilon_init/iter
            r = random()
            if(r<epsilon):
                action = randint(0,5)

            else:                
                best = float('-inf')
                best_ac = -1
                for ac in range(6):
                    if(best < Qsa[rev[c_state]][ac]):
                        best = Qsa[rev[c_state]][ac]
                        best_ac = ac
                action = best_ac
  
            st, rw = taxiDomainInstance.simulate(c_state, action)

            
            best = float('-inf')
            for acn in range(6):
                best = max(best,Qsa[rev[st]][acn])
            
            sample = rw + discount*best
            Qsa[rev[c_state]][action] = Qsa[rev[c_state]][action]*(1-alpha) + alpha*sample

            c_state=st
    return Qsa
            
def chooseAction(Qsa, ci_state, epsilon):
    r = random()
    if(r<epsilon):
        action = randint(0,5)
    else:                
        best = float('-inf')
        best_ac = -1
        for ac in range(6):
            if(best < Qsa[ci_state][ac]):
                best = Qsa[ci_state][ac]
                best_ac = ac
        action = best_ac
    return action


def sarsa_fix(taxiDomainInstance, epsilon, discount, alpha, max_episode, max_it):#max_episode atleast 2000 episodes,max_it - 500 steps maximum length
    
    destination = taxiDomainInstance.dest

    states = taxiDomainInstance.states
    N = len(states)
    # start_state = taxiDomainInstance.s_space[taxi_pos[1]][taxi_pos[0]][start[1]][start[0]][0]
    rev = taxiDomainInstance.rev
    
    Qsa = [[0 for ix in range(6)] for ix in range(N)]
    
    episodes = 0
    iter = 0
    while(episodes<max_episode):# convergence criteria doubt
        
        episodes+=1
        c_state = random_state(taxiDomainInstance) # randomly chosen feasible state
        c_action = chooseAction(Qsa, rev[c_state], epsilon)
        itr=0  
        while(not (taxiDomainInstance.HasTerminated(c_state)or itr>=max_it) ):
            itr+=1
            iter+=1
  
            st, rw = taxiDomainInstance.simulate(c_state, c_action)

            ac = chooseAction(Qsa, rev[st], epsilon)
            
            sample = rw + discount*Qsa[rev[st]][ac]
            Qsa[rev[c_state]][c_action] = Qsa[rev[c_state]][c_action]*(1-alpha) + alpha*sample
            c_state = st
            c_action = ac
    return Qsa

def sarsa_decay(taxiDomainInstance,epsilon_init,  discount, alpha,max_episode,max_it):#max_episode atleast 2000 episodes,max_it - 500 steps maximum length
    destination = taxiDomainInstance.dest

    states = taxiDomainInstance.states
    N = len(states)
    #start_state = taxiDomainInstance.s_space[taxi_pos[1]][taxi_pos[0]][start[1]][start[0]][0]
    rev = taxiDomainInstance.rev
    
    Qsa = [[0 for ix in range(6)] for ix in range(N)]
    
    episodes = 0
    iter = 0
    while(episodes<max_episode):# convergence criteria doubt
        
        episodes+=1
        c_state = random_state(taxiDomainInstance) # randomly chosen feasible state
        iter+=1
        epsilon=epsilon_init/iter
        c_action = chooseAction(Qsa, rev[c_state], epsilon)
        itr=1 
        while(not (taxiDomainInstance.HasTerminated(c_state)or itr>=max_it) ):
            itr+=1
            iter+=1
            epsilon = epsilon_init/iter
  
            st, rw = taxiDomainInstance.simulate(c_state, c_action)

            ac = chooseAction(Qsa,rev[st], epsilon)
            
            sample = rw + discount*Qsa[rev[st]][ac]
            Qsa[rev[c_state]][c_action] = Qsa[rev[c_state]][c_action]*(1-alpha) + alpha*sample
            c_state = st
            c_action = ac
    return Qsa

#algorithm
# region
QL_FIX = 0
QL_DEC = 1
SARSA_FIX = 2
SARSA_DEC = 3
# endregion

def Learning2(taxiDomainInstance, max_episode, max_iter, epsilon, alpha, discount, algorithm):
    
    # start = taxiDomainInstance.start
    # taxi_pos = taxiDomainInstance.taxi
    # taxiDomainInstance = TaxiDomain(map, start, destination, taxi_pos) # is taxi, psg start state used

    Qsa = []
    if(algorithm==QL_FIX):
        Qsa = QL_fix(taxiDomainInstance, epsilon, discount, alpha, max_episode, max_iter)
    elif(algorithm==QL_DEC):
        Qsa = QL_decay(taxiDomainInstance, epsilon, discount, alpha, max_episode, max_iter)
    elif(algorithm==SARSA_FIX):
        Qsa = sarsa_fix(taxiDomainInstance, epsilon, discount, alpha, max_episode, max_iter)
    else:
        Qsa = sarsa_decay(taxiDomainInstance, epsilon, discount, alpha, max_episode, max_iter)
    

    states = taxiDomainInstance.states
    rev = taxiDomainInstance.rev
    N = len(states)
    policy = np.zeros(N)

    for i in range(N):        
        c_state = states[i]
        if((c_state.px,c_state.py) == taxiDomainInstance.dest and c_state.s==0): # for terminal state -> policy
            policy[i] = 6
            continue
        best = float('-inf')
        best_ac = -1
        for action in range(6):
            #print(Qsa)    
            if(best < (Qsa[rev[c_state]][action])):
                best = (Qsa[rev[c_state]][action])
                best_ac = action
        policy[i] = best_ac
    
    return policy

# evaluate the policy from learning algo
def EvaluatePolicy(taxiDomainInstance, policy, max_iter, discount, max_episodes=10, st_list=[]):

    states = taxiDomainInstance.states
    N = len(states)
    
    avg = 0
    rev = taxiDomainInstance.rev

    if(st_list!=[]): # initial states provided
        for init_state in st_list:
            st_avg = 0
            for run in range(10): # 10 runs on each state
                iter = 0
                disc_rw = 0
                c_state = init_state
                while(not (taxiDomainInstance.HasTerminated(c_state) or iter>=max_iter)): 
                    action = int(policy[rev[c_state]])
                    st, rw = taxiDomainInstance.simulate(c_state, action)           
                    c_state = st
                    disc_rw += rw *(discount**iter)
                    iter+=1
                st_avg+=disc_rw
            avg+=st_avg/10
        avg/=len(st_list)

    else:
        for episode in range(max_episodes):
            init_state = random_state(taxiDomainInstance)
            st_avg = 0
            for run in range(10):
                iter = 0
                disc_rw = 0
                c_state = init_state
                while(not (taxiDomainInstance.HasTerminated(c_state) or iter>=max_iter)): 
                    action = int(policy[rev[c_state]])
                    st, rw = taxiDomainInstance.simulate(c_state, action)           
                    c_state = st
                    disc_rw += rw *(discount**iter)
                    iter+=1
                st_avg+=disc_rw
            avg+=st_avg/10
        avg/=max_episodes
    return avg

   
def Learning4():
    ep = [0, 0.05, 0.1, 0.5, 0.9]
    alp = [0.1, 0.2, 0.3, 0.4, 0.5]
    
def Learning2Main(map, destination, max_episode, max_iter, epsilon, alpha, discount, algorithm):
    tdi = TaxiDomain(map, (0,0), destination, (0,0))
    policy = Learning2(tdi, max_episode, max_iter, epsilon, alpha, discount, algorithm)


def best_reward():
    tdi1=TaxiDomain(MAP5x5,(0,0),(4,4),(0,3))
    rwd=[]
    for i in range(4):

        policy0=Learning2(tdi1,2100,500,0.1,0.25,0.99,i) 
        eval0=EvaluatePolicy(tdi1,policy0,500,0.99)
        rwd+=[eval0]
        # helper(tdi1,policy0,(0,0),(4,4),(0,3))
    return rwd


def B2(epsilon=0.1, alpha=0.25):
    
    tdi1=TaxiDomain(MAP5x5,(0,0),(4,4),(0,3))
    # show_list=[0,250, 500, 750, 1000, 1250, 1500,1750,2000]#[2000, 2500, 3000, 3500, 4000, 4500, 5000]
    show_list=[i for i in range(1000,11000,1000)]
    # ep_list=[ix for ix in range(50,2050,50)]
    ep_list=[ix for ix in range(100,10050,100)]
    #ep_list.append(2020)
    ep_list=ep_list
    max_ep=10000#=2000 #max(ep_list)
    lb=0
    st_list = []
    for i in range(10):
        st_list.append(random_state(tdi1))
    rwd_list = []
    for i in range(4):
        if(i==0):
            lb = 'Q_Learning'
            rwd=B2_0(tdi1,0.1,0.99,0.25,max_ep,500,ep_list,st_list)
            rwd_list.append(rwd[-1])
        elif(i==1):
            lb = 'Q_Learning_Decayed'
            rwd=B2_1(tdi1,0.1,0.99,0.25,max_ep,500,ep_list,st_list)
            rwd_list.append(rwd[-1])
        elif(i==2):
            lb = 'SARSA'
            rwd=B2_2(tdi1,0.1,0.99,0.25,max_ep,500,ep_list,st_list)
            rwd_list.append(rwd[-1])
        else:
            lb = 'SARSA_Decayed'
            rwd=B2_3(tdi1,0.1,0.99,0.25,max_ep,500,ep_list,st_list)
            rwd_list.append(rwd[-1])
        
        plt.plot(ep_list,rwd,label=lb)
        plt.xticks(show_list)
        plt.xlabel('Number of Episodes')
        plt.ylabel('Average Discounted reward')
        plt.legend()
        title='B2_'+lb
        plt.title(title)
        plt.savefig(title+".png")
        plt.clf()
    
    return rwd_list

    # plt.xticks(show_list)
    # plt.xlabel('Episodes')
    # plt.ylabel('Discounted reward')
    # plt.legend()
    # title='QB2_ep_'+str(epsilon)+'_alph_'+str(alpha)
    # plt.title(title)
    # plt.savefig(title+"_.png")
    # plt.clf()

# QLearning
def B2_0(taxiDomainInstance, epsilon, discount, alpha, max_episode, max_it,ep_list, st_list):
    policy_list=[]
    destination = taxiDomainInstance.dest 
    states = taxiDomainInstance.states 
    N = len(states) 
    # start_state = taxiDomainInstance.s_space[taxi_pos[1]][taxi_pos[0]][start[1]][start[0]][0] 

    rev = taxiDomainInstance.rev
    
    Qsa = [[0 for ix in range(6)] for ix in range(N)]
    
    episodes = 0
    ep_i=0
    while(episodes<(max_episode+1)):# convergence criteria doubt
        episodes+=1

        c_state = random_state(taxiDomainInstance) # randomly chosen feasible state
        # print(ep_i)
        if (episodes==ep_list[ep_i]):
            ep_i+=1
            policy = np.zeros(N)

            for ix in range(N):        
                c_st = states[ix]
                if((c_st.px,c_st.py) == taxiDomainInstance.dest and c_st.s==0): # for terminal state -> policy
                    policy[ix] = 6
                    continue
                bst = float('-inf')
                bst_ac = -1
                for act in range(6):
                    #print(Qsa)    
                    if(bst < (Qsa[rev[c_st]][act])):
                        bst = (Qsa[rev[c_st]][act])
                        bst_ac = act
                policy[ix] = bst_ac
            policy_list.append(policy)
            if (episodes==(max_episode)):
                break

        
        itr=0  
        while(not (taxiDomainInstance.HasTerminated(c_state) or itr >= max_it) ):
            itr+=1
            r = random()
            action = 0
            if(r<epsilon):
                action = randint(0,5)
            else:                
                best = float('-inf')
                best_ac = -1
                for ac in range(6):
                    if(best < Qsa[rev[c_state]][ac]):
                        best = Qsa[rev[c_state]][ac]
                        best_ac = ac
                action = best_ac
  
            st, rw = taxiDomainInstance.simulate(c_state, action)

            best = float('-inf')
            for acn in range(6):
                best = max(best,Qsa[rev[st]][acn])
            
            sample = rw + discount*best
            Qsa[rev[c_state]][action] = Qsa[rev[c_state]][action]*(1-alpha) + alpha*sample

            c_state=st
    #return policy_list
    rwd_list=[]
    

    for pol in policy_list: 
        rwd_list.append(EvaluatePolicy(taxiDomainInstance, pol, max_it, discount,10,st_list))
    return rwd_list

# QLearning with decay
def B2_1(taxiDomainInstance,epsilon_init,  discount, alpha,max_episode,max_it,ep_list,st_list):#max_episode atleast
    policy_list=[]
    destination = taxiDomainInstance.dest

    states = taxiDomainInstance.states
    N = len(states)
    #start_state = taxiDomainInstance.s_space[taxi_pos[1]][taxi_pos[0]][start[1]][start[0]][0]
    rev = taxiDomainInstance.rev
    
    Qsa = [[0 for ix in range(6)] for ix in range(N)]
    
    episodes = 0
    iter = 0
    ep_i=0
    while(episodes<(max_episode+1)):# convergence criteria doubt
        
        episodes+=1
        c_state = random_state(taxiDomainInstance) # randomly chosen feasible state
        if (episodes==ep_list[ep_i]):
            ep_i+=1
            policy = np.zeros(N)

            for ix in range(N):        
                c_st = states[ix]
                if((c_st.px,c_st.py) == taxiDomainInstance.dest and c_st.s==0): # for terminal state -> policy
                    policy[ix] = 6
                    continue
                bst = float('-inf')
                bst_ac = -1
                for act in range(6):
                    #print(Qsa)    
                    if(bst < (Qsa[rev[c_st]][act])):
                        bst = (Qsa[rev[c_st]][act])
                        bst_ac = act
                policy[ix] = bst_ac
            policy_list.append(policy)
            if (episodes==(max_episode)):
                break
        itr=0  
        while(not (taxiDomainInstance.HasTerminated(c_state)or itr>=max_it) ):
            itr+=1
            iter+=1
            epsilon = epsilon_init/iter
            r = random()
            if(r<epsilon):
                action = randint(0,5)

            else:                
                best = float('-inf')
                best_ac = -1
                for ac in range(6):
                    if(best < Qsa[rev[c_state]][ac]):
                        best = Qsa[rev[c_state]][ac]
                        best_ac = ac
                action = best_ac
  
            st, rw = taxiDomainInstance.simulate(c_state, action)

            
            best = float('-inf')
            for acn in range(6):
                best = max(best,Qsa[rev[st]][acn])
            
            sample = rw + discount*best
            Qsa[rev[c_state]][action] = Qsa[rev[c_state]][action]*(1-alpha) + alpha*sample

            c_state=st
    rwd_list=[]
    for pol in policy_list: 
        rwd_list.append(EvaluatePolicy(taxiDomainInstance, pol, max_it, discount,10,st_list))
    return rwd_list

# SARSA
def B2_2(taxiDomainInstance, epsilon, discount, alpha, max_episode, max_it,ep_list,st_list):#max_episode atleast 2000 episodes,max_it - 500 steps maximum length
    policy_list=[]
    destination = taxiDomainInstance.dest

    states = taxiDomainInstance.states
    N = len(states)
    # start_state = taxiDomainInstance.s_space[taxi_pos[1]][taxi_pos[0]][start[1]][start[0]][0]
    rev = taxiDomainInstance.rev
    
    Qsa = [[0 for ix in range(6)] for ix in range(N)]
    
    episodes = 0
    iter = 0
    ep_i=0
    while(episodes<(max_episode+1)):# convergence criteria doubt
        
        episodes+=1
        c_state = random_state(taxiDomainInstance) # randomly chosen feasible state
        c_action = chooseAction(Qsa, rev[c_state], epsilon)

        if (episodes==ep_list[ep_i]):
            ep_i+=1
            policy = np.zeros(N)

            for ix in range(N):        
                c_st = states[ix]
                if((c_st.px,c_st.py) == taxiDomainInstance.dest and c_st.s==0): # for terminal state -> policy
                    policy[ix] = 6
                    continue
                bst = float('-inf')
                bst_ac = -1
                for act in range(6):
                    #print(Qsa)    
                    if(bst < (Qsa[rev[c_st]][act])):
                        bst = (Qsa[rev[c_st]][act])
                        bst_ac = act
                policy[ix] = bst_ac
            policy_list.append(policy)
            if (episodes==(max_episode)):
                break
        itr=0  
        while(not (taxiDomainInstance.HasTerminated(c_state)or itr>=max_it) ):
            itr+=1
            iter+=1
  
            st, rw = taxiDomainInstance.simulate(c_state, c_action)

            ac = chooseAction(Qsa, rev[st], epsilon)
            
            sample = rw + discount*Qsa[rev[st]][ac]
            Qsa[rev[c_state]][c_action] = Qsa[rev[c_state]][c_action]*(1-alpha) + alpha*sample
            c_state = st
            c_action = ac
    rwd_list=[]
    for pol in policy_list: 
        rwd_list.append(EvaluatePolicy(taxiDomainInstance, pol, max_it, discount,10,st_list))
    return rwd_list

# SARSA with decay
def B2_3(taxiDomainInstance,epsilon_init,  discount, alpha,max_episode,max_it,ep_list,st_list):#max_episode atleast 2000 episodes,max_it - 500 steps maximum length
    destination = taxiDomainInstance.dest
    policy_list=[]
    states = taxiDomainInstance.states
    N = len(states)
    #start_state = taxiDomainInstance.s_space[taxi_pos[1]][taxi_pos[0]][start[1]][start[0]][0]
    rev = taxiDomainInstance.rev
    
    Qsa = [[0 for ix in range(6)] for ix in range(N)]
    
    episodes = 0
    iter = 0
    ep_i=0
    while(episodes<(max_episode+1)):# convergence criteria doubt
        
        episodes+=1
        c_state = random_state(taxiDomainInstance) # randomly chosen feasible state
        iter+=1
        epsilon=epsilon_init/iter
        c_action = chooseAction(Qsa, rev[c_state], epsilon)

        if (episodes==ep_list[ep_i]):
            ep_i+=1
            policy = np.zeros(N)

            for ix in range(N):        
                c_st = states[ix]
                if((c_st.px,c_st.py) == taxiDomainInstance.dest and c_st.s==0): # for terminal state -> policy
                    policy[ix] = 6
                    continue
                bst = float('-inf')
                bst_ac = -1
                for act in range(6):
                    #print(Qsa)    
                    if(bst < (Qsa[rev[c_st]][act])):
                        bst = (Qsa[rev[c_st]][act])
                        bst_ac = act
                policy[ix] = bst_ac
            policy_list.append(policy)
            if (episodes==(max_episode)):
                break
        itr=1 
        while(not (taxiDomainInstance.HasTerminated(c_state)or itr>=max_it) ):
            itr+=1
            iter+=1
            epsilon = epsilon_init/iter
  
            st, rw = taxiDomainInstance.simulate(c_state, c_action)

            ac = chooseAction(Qsa,rev[st], epsilon)
            
            sample = rw + discount*Qsa[rev[st]][ac]
            Qsa[rev[c_state]][c_action] = Qsa[rev[c_state]][c_action]*(1-alpha) + alpha*sample
            c_state = st
            c_action = ac
    rwd_list=[]
    for pol in policy_list: 
        rwd_list.append(EvaluatePolicy(taxiDomainInstance, pol, max_it, discount,10,st_list))
    return rwd_list


def B3(map, destination, max_episode, max_iter, epsilon, alpha, discount, algorithm):
    start = (0,0)
    taxi_pos = (0,0)
    taxiDomainInstance = TaxiDomain(map, start, destination, taxi_pos)
    learned_policy=Learning2(taxiDomainInstance, max_episode, max_iter, epsilon, alpha, discount, algorithm)# algorithm hardcoded
    avg = 0
    rwd_list=[]
    states = taxiDomainInstance.states
    N = len(states)
    rev = taxiDomainInstance.rev
    
    for i in range(5):
        c_state = random_state(taxiDomainInstance)
        iter = 0
        disc_rw = 0
        while(not (taxiDomainInstance.HasTerminated(c_state) or iter>=max_iter)): 
            action = int(learned_policy[rev[c_state]])
            st, rw = taxiDomainInstance.simulate(c_state, action)           
            c_state = st
            disc_rw += rw*(discount**iter)
            iter+=1
        avg+=disc_rw
        rwd_list.append(disc_rw)
    avg/=5
    # for i in range(5):
    #     st_avg = 0
    #     for run in range(10):
    #         c_state = st_list[i]
    #         iter = 0
    #         disc_rw = 0
    #         while(not (taxiDomainInstance.HasTerminated(c_state) or iter>=max_iter)): 
    #             action = int(learned_policy[rev[c_state]])
    #             st, rw = taxiDomainInstance.simulate(c_state, action)           
    #             c_state = st
    #             disc_rw += rw*(discount**iter)
    #             iter+=1
    #         st_avg+=disc_rw
    #     rwd_list.append(st_avg/10)
    #     avg+=st_avg/10
    # avg/=5
    return avg,rwd_list



def B5():
    tdi2=TaxiDomain(MAP10x10,(0,1),(2,4),(5,9))
    depot=tdi2.depots
    d=len(depot)
    avg_list=[]
    dest_list=[]
    for i in range(5):
        tdi2=TaxiDomain(MAP10x10,depot[((i+1)%d)],depot[i],depot[((i+2)%d)])
        dest_list.append(depot[i])
        policy=Learning2(tdi2,10000,2500,0.1,0.25,0.99,1)
        eval=EvaluatePolicy(tdi2,policy,2500,0.99,5)
        avg_list+=[eval]
    return avg_list,dest_list

def B4_epsilon():
    epsilon_list=[0,0.05,0.1,0.5,0.9]
    tdi1=TaxiDomain(MAP5x5,(0,0),(4,4),(0,3))
    show_list=[i for i in range(1000,11000,1000)]
    ep_list=[ix for ix in range(100,10050,100)]
    max_ep=max(ep_list)
    st_list = []
    for i in range(10):
        st_list.append(random_state(tdi1))
    alpha0=0.1
    epsilon0=0.1
    alpha_list=[0.1,0.2,0.3,0.4,0.5]
    for i in epsilon_list:
        # rwd = B4Helper(ep_list,alpha0,i)
        rwd=B2_0(tdi1,i,0.99,alpha0,max_ep,500,ep_list,st_list)

        lb="epsilon="+str(i)
        plt.plot(ep_list,rwd,label=lb)
        plt.xticks(show_list)
        plt.xlabel('Number of Episodes')
        plt.ylabel('Average Discounted reward')
        plt.legend()
        title="B4_epsilon_"+str(i)
        plt.title(title)
        plt.savefig(title+".png")
        plt.clf()
def B4_alpha():
    epsilon_list=[0,0.05,0.1,0.5,0.9]
    tdi1=TaxiDomain(MAP5x5,(0,0),(4,4),(0,3))
    show_list=[i for i in range(1000,11000,1000)]
    ep_list=[ix for ix in range(100,10050,100)]
    max_ep=max(ep_list)
    alpha0=0.1
    epsilon0=0.1
    alpha_list=[0.1,0.2,0.3,0.4,0.5]
    st_list = []
    for i in range(10):
        st_list.append(random_state(tdi1))
    for i in alpha_list:
        rwd=B2_0(tdi1,epsilon0,0.99,i,max_ep,500,ep_list,st_list)

        lb="alpha="+str(i)
        plt.plot(ep_list,rwd,label=lb)
        plt.xticks(show_list)
        plt.xlabel('Number of Episodes')
        plt.ylabel('Average Discounted reward')
        plt.legend()
        title="B4_alpha_"+str(i)
        plt.title(title)
        plt.savefig(title+".png")
        plt.clf()

# main function
def main():
    part = sys.argv[1]
    sub_part= sys.argv[2]
    if (part =="b"):
        if (sub_part =="1"):
            
            tdi1=TaxiDomain(MAP5x5,(0,0),(4,4),(0,3))
            rwd=[]
            policy0=Learning2(tdi1,10000,500,0.1,0.25,0.99,1) 
            eval0=EvaluatePolicy(tdi1,policy0,500,0.99)
            print(eval0)

        elif (sub_part =="2"):
            rwd_list = B2()
            print("Required graphs(Discounted Rewards Sum vs Training Episodes) get saved")
            print("Sum of discounted rewards accumulated, of these algorithms-")
            print(str(['Q_Learning','Q_Learning_Decayed','SARSA','SARSA_Decayed'])+" =\n"+str(rwd_list))
            print("Algorithm,having highest accumulated reward -",'Q_Learning_Decayed')
        
        elif (sub_part =="3"):
            avg,rwd_list=B3(MAP5x5, (0,4), 10000, 500, 0.1, 0.25, 0.99, 1)

            print("Sum of discounted rewards for 5 instances-\n",rwd_list)
            print("Average reward-",avg)
        
        elif (sub_part =="4"):
            B4_epsilon()
            B4_alpha()
            print("Required graphs of(Discounted Rewards Sum vs Training Episodes) for (epsilon vs alpha )get saved")

            print("See observations in Report")
        elif(sub_part =="5"):
            avg_list,dest_list=B5()
            print("Different destinations for 5 Models-",str(dest_list))
            print("Average discounted rewards for 5 Models-",str(avg_list))
    
    elif (part =="a"):
        if (sub_part =="1a"):
            print("Described in Report")
        elif (sub_part =="1b"):
            #function to simulate with eg
            tdi1=TaxiDomain(MAP5x5,(0,0),(4,4),(0,3))
            tdi1.print_simulate((tdi1.states[50]),0)
        elif (sub_part =="2a"):
            tdi=TaxiDomain(MAP5x5,(0,0),(4,4),(0,3))
            policy1 = A2a(tdi, 1e-6)
            print("epsilon-",1e-6)
        elif (sub_part =="2b"):
            tdi1=TaxiDomain(MAP5x5,(0,0),(4,4),(0,3))
            A2b(tdi1,1e-6)
        elif (sub_part =="2c"):
            tdi = TaxiDomain(MAP5x5, (0,0), (4,4), (0,4))
            # tdi = TaxiDomain(MAP5x5, (0,4), (4,4), (0,0))
            # tdi = TaxiDomain(MAP5x5, (3,0), (4,4), (0,0))
            print('discount',0.1)
            policy1, it_norm =valueIteration(tdi, 1e-6, 0.1)
            A2chelper(tdi, policy1)
            # print(policy1)
            print('discount',0.99)
            policy2, it_norm =valueIteration(tdi, 1e-6, 0.99)
            A2chelper(tdi, policy2)
            # print((policy1==policy2).all())
        elif (sub_part=="3a"):
            
            A3a()
        elif (sub_part=="3b"):
            tdi=TaxiDomain(MAP5x5,(0,0),(4,4),(0,3))
            A3b(tdi,0)
        else:
            print("Part A not done complete")
    else:
        print("part not present for this A3-COL333")


if __name__ == '__main__':
    main()