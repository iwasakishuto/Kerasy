# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from ..utils import flush_progress_bar
from ..utils import galleryplot

class Pannel():
    n=None;m=None;digit=None
    goal=None

    def __init__(self,string,par_string,h,heuristic_method="Manhattan_distance",n=None,m=None,goal_str=None):
        """
        @param string           : (str) encoded pannel states.
        @param par_string       : (str) encoded parent pannel states. It is neccesary for traceback.
        @param h                : (int) Transition counts from initial states.
        @param heuristic_method : (str) How to estimate the total transition costs.
        @param n,m              : (int) pannel size (n × m)
        @param goal_str         : (str) encoded last pannel states.
        """
        if n is not None:
            Pannel.n = n
            Pannel.m = m
            Pannel.digit = len(str(n*m))
            Pannel.goal = goal_str

        self.str = string
        self.h = h
        self.hm = heuristic_method
        self.g = self.heuristic(heuristic_method)
        self.par = par_string

    def __lt__(self, other):
        """ this enables us to compare the pannel directoly. """
        return (self.h+self.g)<(other.h+other.g)

    @staticmethod
    def arr2str(array):  return ",".join(array.flatten().astype(str))

    @staticmethod
    def str2arr(string): return np.asarray([int(e) for e in string.split(",")]).reshape(Pannel.n,Pannel.m)

    def swap(self,array,xi,yi,xj,yj):
        """ array[xi,yi] ↔︎ array[xj,yj] """
        array = np.copy(array)
        tmp = array[xi,yi]
        array[xi,yi]=array[xj,yj]
        array[xj,yj]=tmp
        return self.arr2str(array) # NOTE: type(return) is str.

    def heuristic(self, method):
        """ Manhattan distance """
        if method=="Manhattan_distance":
            g=0
            goal_arr = self.str2arr(Pannel.goal)
            arr = self.str2arr(self.str)
            for i in range(1,Pannel.n*Pannel.m+1):
                idx = np.argmax(arr==i)
                x,y = (idx//Pannel.m, idx%Pannel.m)
                goal_idx = np.argmax(goal_arr==i)
                goal_x,goal_y = (goal_idx//Pannel.m, goal_idx%Pannel.m)
                g += abs(goal_x-x)+abs(goal_y-y)

        elif method=="Absolute_distance":
            g = sum([ei!=ej for ei,ej in zip(self.str.split(","), Pannel.goal.split(","))])
        else:
            raise KeyError(f"{method} is not defined.")
        return g

    def transibles(self):
        arr = self.str2arr(self.str)
        idx = np.argmax(arr==-1)
        x,y = (idx//Pannel.m, idx%Pannel.m)

        cands=[]
        if x!=0: cands.append(self.swap(arr,x,y,x-1,y))
        if y!=0: cands.append(self.swap(arr,x,y,x,y-1))
        if x!=Pannel.n-1: cands.append(self.swap(arr,x,y,x+1,y))
        if y!=Pannel.m-1: cands.append(self.swap(arr,x,y,x,y+1))

        return [Pannel(string=c_str, par_string=self.str, h=self.h+1, heuristic_method=self.hm) for c_str in cands]

    def plot(self, ax=None):
        if ax==None:
            fig,ax=plt.subplots()

        arr = self.str2arr(self.str)
        matrix= np.asarray(arr==-1, dtype=int)
        ax.matshow(matrix, alpha=0.3, vmin=0, vmax=1,cmap="Greys")
        for i in range(Pannel.n):
            for j in range(Pannel.m):
                text = "" if arr[i][j]<0 else arr[i][j]
                ax.text(x=j, y=i, s=text, va='center', ha='center')
        ax.set_xticks([]), ax.set_yticks([])
        return ax

def OptimalTransit(n_rows, n_cols, initial_str, last_str,
                   heuristic_method="Manhattan_distance", max_iter=100,
                   verbose=1, n_fig_cols=5, retval=False):
    initial_state = Pannel(
        string=initial_str,
        par_string=None,
        h=0,
        heuristic_method=heuristic_method,
        n=n_rows,
        m=n_cols,
        goal_str=last_str,
    )

    open_list = [initial_state]
    closed_list = {}
    for it in range(max_iter):
        if len(open_list) == 0: break
        state = open_list.pop(np.argmin(open_list))
        closed_list[state.str] = state
        if state.str==last_str: break
        open_list += [s for s in state.transibles() if s.str not in closed_list]
        flush_progress_bar(it, max_iter, metrics={"transition" : state.h}, verbose=verbose)

    if last_str not in closed_list:
        print("Can't reach to the last state.")
    else:
        n_transition = closed_list[last_str].h
        pannel_str = last_str
        pannel_lst = []
        # Trace Back.
        while pannel_str:
            state = closed_list[pannel_str]
            pannel_str = state.par
            pannel_lst.append(state)

        if retval:
            return pannel_lst

        else:
            def plot_state_transition(ax, i, state):
                state.plot(ax=ax)
                ax.set_title(f"No.{i:<0{len(str(n_transition))}}", fontsize=14)
                return ax

            fig, axes = galleryplot(
                func=plot_state_transition,
                argnames=["i", "state"],
                iterator=enumerate(pannel_lst),
                ncols=n_fig_cols,
            )
            plt.tight_layout()
            plt.show()
