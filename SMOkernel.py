import os
import numpy as np
import pandas as pd



os.chdir("C:\\Users\\cimlab\\Desktop\\SMO_kernel")
trainset = pd.read_csv("alphabet_DU_training_header.csv")

testset = pd.read_csv("alphabet_DU_testing_header.csv")



#data preprocessing
for i in range(0,len(trainset.index)):
    if trainset["class"][i]==4:
        trainset.set_value(i,"class",1)
    elif trainset["class"][i]==21:
        trainset.set_value(i,"class",-1)

for i in range(0,len(testset.index)):
    if testset["class"][i]==4:
        testset.set_value(i,"class",1)
    elif testset["class"][i]==21:
        testset.set_value(i,"class",-1)

#Define the kernel function
def kernel(para,w1,w2,istar = -1,jstar = -1):
    if istar == -1:
        x1 = w1[:]
    else:
        x1 = w1[istar]
    if jstar == -1:
        x2 = w2[:]
    else:
        x2 = w2[jstar]
    if Ktype == 1:
        return x1*x2.T
    elif Ktype == 2:
        return np.power(1*(x1*x2.T),para)
    elif Ktype == 3:
        new = np.power(x1 - x2,2)
        return np.exp(-sum(new.T).T/(2*para**2))
    


#Cross Validation
############
def cross_c(n,train,eps):

    m = len(train.index)



    size = round(m/n)

    fold=[]

    all = list(range(0,m))
    fold = []
    f = []
    for i in range(0,n-1):
        f = []
        for j in range(0,size):
            temp = all.pop(int(np.random.randint(0,len(all))))
            f.append(int(temp))
        fold.append(f)

    fold.append(all)


    low = 1.
    up = 11.

    cbest = 0.
    r_matrix = [0]*11
    risk = 0.
    for i in range(0,3):
        cut = 10**-i
        C_range = np.array(np.linspace(low,up,11))
        count = 0
        for j in C_range:
            print("Cross Validating C, Please keep waiting. . .")
            print("C now:",j)
        
            for k in range(0,n):
                all=list(range(0,m))
                for l in np.array(fold[k]):
                    all.remove(l)
                if Ktype == 3:
                    temp = 100
                else:
                    temp = 2
                ris,a,b,g,d,e,f,o,x,z = SMO(train.iloc[all,:],train.iloc[fold[k],:],j,eps,temp)
                risk = risk+ris
            risk = risk/n
            r_matrix[count] = risk
            count += 1
        amin = np.argmin(r_matrix)
        print(r_matrix)
        cbest = C_range[amin]
        low = low + cut*(amin-0.5)
        up = low + cut

    return round(cbest,2)
###############

def cross_kernel(n,train,eps,cbest):

    m = len(train.index)



    size = round(m/n)

    fold=[]

    all = list(range(0,m))
    fold = []
    f = []
    for i in range(0,n-1):
        f = []
        for j in range(0,size):
            temp = all.pop(int(np.random.randint(0,len(all))))
            f.append(int(temp))
        fold.append(f)

    fold.append(all)

    if Ktype == 2:
        low = 1.
        up = 11.

    elif Ktype == 3:
        low = 100.
        up = 1000.


    r_matrix = [0]*11
    risk = 0.
    
    cut = 10
    d_range = np.array(np.linspace(low,up,11))
    count = 0
    for j in d_range:
        if Ktype == 2:
            print("Cross Validating d, Please keep waiting. . .")
            print("d now:",j)
        elif Ktype == 3:
            print("Cross Validating sigma, Please keep waiting. . .")
        print("sigma now:",j)
        
        for k in range(0,n):
            all=list(range(0,m))
            for l in np.array(fold[k]):
                all.remove(l)
            ris,a,b,g,d,e,f,o,q,l = SMO(train.iloc[all,:],train.iloc[fold[k],:],cbest,eps,j)
            risk = risk+ris
        risk = risk/n
        r_matrix[count] = risk
        count += 1
    amin = np.argmin(r_matrix)
    print(r_matrix)
    dbest = d_range[amin]

    return round(dbest)





#SMO algorithm
#####################
def SMO(train,test,C,eps,para = 0):

#parametrs setting
    paraC = C
    tol = eps
    train.index = range(0,len(train.index))
    test.index = range(0,len(test.index))
#Initialization
    m = len(train.index)
    t = 1
    j = 0
    kkt = 0
    y = np.repeat(0.,m)
    E = np.repeat(0.,m)
    F = np.repeat(0.,m)
    c = np.repeat(0.,m)

    I0 = []
    I1 = []
    I2 = []
    I3 = []
    I4 = []
    for i in range(0,m):
        y[i] = train["class"][i]
        F[i] = -train["class"][i]
        if y[i] == 1:
            I1.append(i)
        elif y[i] == -1:
            I4.append(i)

    x = np.matrix(train.iloc[:,1:m])

    L = np.repeat(0.,m)
    L_old = np.repeat(0.,m)
    b = 0.
    Bup = -1.
    Blow = 1.

    examALL = True

    while kkt == 0:

        if examALL == True:
            Iout = 0
            if len(I0) > 0 and examALL == False:

                for j in np.array(I0):
                    F[j] = float(sum(L[:]*y[:]*kernel(para,x,x,-1,j))-y[j])
                    if i != j and Bup+tol<F[j]<Blow-tol:
                        Iout = I0
                        break
            else:
                Iout = range(0,m)
            
        for i in Iout:
            c[i] = sum(L[:]*y[:]*kernel(para,x,x,-1,i)) + b
            E[i] = c[i]-y[i]
            
            if examALL == True:#First pass
                tp = 0
                tvalue = 0
                for k in range(0,m):
                    c[k] = sum(L[:]*y[:]*kernel(para,x,x,-1,k)) + b
                    E[k] = c[i]-y[i]
                    if i != k and Bup+tol<F[k]<Blow-tol and abs(E[k]-E[i])>tvalue:
                        tvalue = abs(E[k]-E[i])
                        tp = k
                j = tp
                i,j,F,L_old,L,Bup,Blow,kkt,I0,I1,I2,I3,I4,x,y,paraC,m,tol,E,b,c = step(i,j,F,L_old,L,Bup,Blow,kkt,I0,I1,I2,I3,I4,x,y,paraC,m,tol,E,b,c,para)

                if False:
                    I = list(range(0,m))
                    np.random.shuffle(I)
                    for j in I:
                        
                        if i!=j:
                
                            i,j,F,L_old,L,Bup,Blow,kkt,I0,I1,I2,I3,I4,x,y,paraC,m,tol,E,b,c = step(i,j,F,L_old,L,Bup,Blow,kkt,I0,I1,I2,I3,I4,x,y,paraC,m,tol,E,b,c,para)
            else:#after first pass
                temp = 0
                if (i in list(set(I0) or set(I1) or set(I2))) and F[i]<Blow-tol:
                    temp = 1
                elif (i in list(set(I0) or set(I3) or set(I4))) and F[i]>Bup+tol:
                    temp = 1

                if temp == 1:
                    
                    I = 0
                    if len(I0) > 0:
                        
                        for j in np.array(I0):
                            F[j] = float(sum(L[:]*y[:]*(kernel(para,x,x,-1,j)))-y[j])
                            if i != j and Bup+tol<F[j]<Blow-tol:
                                I = I0
                                break
                    if I == 0:
                        I = range(0,m)
                    I = list(I)
                    tp = 0
                    tvalue = 0
                    for k in I:
                        c[k] = sum(L[:]*y[:]*kernel(para,x,x,-1,k)) + b
                        E[k] = c[i]-y[i]
                        if i != k and (((k in list(set(I0) or set(I1) or set(I2))) and F[k]<Blow-tol)or((j in list(set(I0) or set(I3) or set(I4))) and F[k]>Bup+tol)) and abs(E[k]-E[i])>tvalue:
                            tvalue = abs(E[k]-E[i])
                            tp = k
                    j = tp
                    i,j,F,L_old,L,Bup,Blow,kkt,I0,I1,I2,I3,I4,x,y,paraC,m,tol,E,b,c = step(i,j,F,L_old,L,Bup,Blow,kkt,I0,I1,I2,I3,I4,x,y,paraC,m,tol,E,b,c,para)
                    np.random.shuffle(I)
                    for j in I:
                        temp = 0
                        if (j in list(set(I0) or set(I1) or set(I2))) and F[j]<Blow-tol :
                            temp = 1
                        elif (j in list(set(I0) or set(I3) or set(I4))) and F[j]>Bup+tol :
                            temp = 1

                        if i == j:
                            temp = 0

                        if temp == 1:
                            i,j,F,L_old,L,Bup,Blow,kkt,I0,I1,I2,I3,I4,x,y,paraC,m,tol,E,b,c = step(i,j,F,L_old,L,Bup,Blow,kkt,I0,I1,I2,I3,I4,x,y,paraC,m,tol,E,b,c,para)
                            if kkt ==1:
                                break
        if examALL == True:
            examALL = False

    print("converge!")
    #Calulate Hypothesis
    if len(I0)>0:
        sp = I0[0]
    else:
        sp = 0
    bsvm = b

    wsvm = sum(L[:]*y[:]*x[:])


    m = len(test.index)
    xi = np.matrix(test.iloc[:,1:len(test.columns)])


    h = np.array(np.repeat(0.,m))
    hsvm = np.repeat(0.,m)
    for i in range(0,m):
        h[i] = np.array(sum(L[:]*y[:]*kernel(para,x,xi,-1,i)).T+bsvm)

    for i in range(0,m):
        if h[i]>0:
            hsvm[i] = 1
        else:
            hsvm[i] = -1

    miss = 0

    y = np.repeat(0.,m)
    for i in range(0,m):
        y[i] = test["class"][i]
    for i in range(0,m):
        if y[i] != hsvm[i]:
            miss += 1

    miss = miss/m
    return miss,bsvm,L,hsvm,F,Bup,Blow,wsvm,c,x
    #################
#Updating rule
def step(i,j,F,L_old,L,Bup,Blow,kkt,I0,I1,I2,I3,I4,x,y,paraC,m,tol,E,b,c,para):
    rho = L[i]+y[i]*y[j]*L[j]

    c[j] = sum(L[:]*y[:]*kernel(para,x,x,-1,j)) + b
    E[j] = c[j]-y[j]
    F[i] = float(sum(L[:]*y[:]*(kernel(para,x,x,-1,i)))-y[i])
    F[j] = float(sum(L[:]*y[:]*(kernel(para,x,x,-1,j)))-y[j])
    eta = -2*kernel(para,x,x,i,j)+kernel(para,x,x,i,i)+kernel(para,x,x,j,j)

    L_old[j] = L[j]

    if y[i] == y[j]:
        Lb = max(0,rho-paraC)
        Hb = min(paraC,rho)
    else:
        Lb = max(0,-rho)
        Hb = min(paraC,paraC-rho)

    if eta>0:
        L[j] = L[j]+y[j]*(E[i]-E[j])/eta
        if bool(L[j] > Hb):
            L[j] = Hb
        elif bool(L[j] < Lb):
            L[j] = Lb
    else:
        Lobj = y[j]*(F[i]-F[j])*Lb
        Hobj = y[j]*(F[i]-F[j])*Hb
        if Lobj<Hobj:
            L[j] = Lobj
        elif Lobj>Hobj:
            L[j] = Hobj
    L_old[i] = L[i]
    if abs(L[j]-L_old[j])>=tol*(L[j]+L_old[j]+tol):
        L[i] = L[i]+y[i]*y[j]*(L_old[j]-L[j])

    b1 = b-E[i]-y[i]*(L[i]-L_old[i])*(kernel(para,x,x,i,i))-y[j]*(L[j]-L_old[j])*kernel(para,x,x,i,j)
    b2 = b-E[j]-y[i]*(L[i]-L_old[i])*(kernel(para,x,x,i,j))-y[j]*(L[j]-L_old[j])*kernel(para,x,x,j,j)

    if 0<L[i]<paraC:
        b = b1
    elif 0<L[j]<paraC:
        b = b2
    else:
        b = (b1+b2)/2

    c[i] = sum(L[:]*y[:]*kernel(para,x,x,-1,i)) + b
    E[i] = c[i]-y[i]
    c[j] = sum(L[:]*y[:]*kernel(para,x,x,-1,j)) + b
    E[j] = c[j]-y[j]

    I0 = []
    I1 = []
    I2 = []
    I3 = []
    I4 = []

    for k in range(0,m):
        if 0<L[k]<paraC:
            I0.append(k)
        elif y[k]==1:
            if L[k]==0:
                I1.append(k)
            elif L[k]==paraC:
                I3.append(k)
        else:
            if L[k]==paraC:
                I2.append(k)
            elif L[k]==0:
                I4.append(k)

    for k in range(0,m):
        F[k] = float((L[:]*y[:]*kernel(para,x,x,-1,k))-y[k])

    Bup = min( np.array(pd.DataFrame(F).iloc[list(set(I0) or set(I1) or set(I2))]) )
    Blow = max( np.array(pd.DataFrame(F).iloc[list(set(I0) or set(I3) or set(I4))]) )

    kkt = int(Bup+2*tol >= Blow)
    return i,j,F,L_old,L,Bup,Blow,kkt,I0,I1,I2,I3,I4,x,y,paraC,m,tol,E,b,c



##########################
#Main Frame

###Input:
print("Please Give the kernel type(1:linear, 2:polynomial, 3:gaussian):")

Ktype = int(input())
while Ktype<=0 or Ktype>3:
    print("kernel can only be 1, 2 or 3:")
    Ktype = int(input())
print("Please Give the tolerance epsilon you can accept(0~1):")
epsilon = input()
epsilon = float(epsilon)
while epsilon<0 or epsilon>1:
    print("Epsilon can only be within 0~1:")
    epsilon = float(input())

print("Do you want to do n-fold cross validation?(Y/N)")
ans = input().lower()
while ans != "y" and ans != "n":
    print("Y or N only!")
    ans = input().lower()
if ans == "y":
    print("Please Give the Parameter of Cross Validation n:")
    n_cross = input()
    n_cross = int(n_cross)
    C_smo = cross_c(n_cross,trainset,epsilon)
    if Ktype !=1:
        d_smo = cross_kernel(n_cross,trainset,epsilon,C_smo)
    else:
        d_smo = 0

else:
    print("Please Give the C value:")
    C_smo = float(input())
    if Ktype == 2:
        print("Please Give the d value:")
        d_smo = int(input())
    elif Ktype == 3:
        print("Please Give the sigma value:")
        d_smo = float(input())
    else:
        d_smo = 0
##

miss,bsvm,L,hsvm,F,Bup,Blow,wsvm,c,x = SMO(trainset,testset,C_smo,epsilon,d_smo)

bi = np.repeat(np.nan,len(L))
output = pd.DataFrame(np.column_stack((trainset,L,bi)))
for i in range(0,len(L)):
    if L[i]>0 and L[i]<C_smo:
        output.iloc[i,6] = c[i]-sum(L[:]*c[:]*kernel(d_smo,x,x,-1,i))
o1 = output
output.dropna()
b = output.iloc[:,len(output.columns)-1]
bmean = np.mean(b)
bstd = np.std(b)

print("Exit(1) or Show data(2) or Save detail result as csv file(3)?")
answer = int(input())
while answer != 1:
    if answer != 2 and answer != 3:
        print("1 or 2 or 3only:")
        answer = int(input())
    elif answer ==2:
        print("Cbest = ",C_smo)
        if Ktype == 2:
            print("d best = ",d_smo)
        elif Ktype == 3:
            print("sigma best = ",d_smo)
        print("Accuracy = ",1-miss)
        print("bsvm = ",bsvm)
        print("wsvm = ",wsvm)
        print("mean bsvm = ",bmean)
        print("standard deviation of bsvm",bstd)
        print("Exit(1) or Show data(2) or Save detail result as csv file(3)?")
        answer = int(input())
    elif answer == 3:
        o1.to_csv("SVM_hypothesis_header.csv")
        print("Finish saving!")
        print("Exit(1) or Show data(2) or Save detail result as csv file(3)?")
        answer = int(input())






            
                
    
