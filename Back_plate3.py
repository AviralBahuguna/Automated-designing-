import tkinter as tk
import numpy as np
from stl import mesh
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from decimal import *
from PIL import Image



getcontext().prec = 10
def calt(q,a,b,E,y,B,imax,jmax):
    # Define the 8 vertices of the cube
    q=(q*9.81)/(a*b)
    print(q)
    t=np.round(math.pow(((q*math.pow(b,4)*0.142)/(E*y*(2.21*math.pow((a/b),3)+1)))*16.32,-1/3),decimals=0) +1
    vertices = np.array([\
                         [-1*b, -1*a, -1*t],
                         [+1*b, -1*a, -1*t],
                         [+1*b, +1*a, -1*t],
                         [-1*b, +1*a, -1*t],
                         [-1*b, -1*a, +1*t],
                         [+1*b, -1*a, +1*t],
                         [+1*b, +1*a, +1*t],
                         [-1*b, +1*a, +1*t]])
        # Define the 12 triangles composing the cube
    faces = np.array([
                      [0,3,1],
                      [1,3,2],
                      [0,4,7],
                      [0,7,3],
                      [4,5,6],
                      [4,6,7],
                      [5,1,2],
                      [5,2,6],
                      [2,3,6],
                      [3,7,6],
                      [0,1,5],
                      [0,5,4]])
        # Create the mesh
    cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            cube.vectors[i][j] = vertices[f[j],:]
                
                
            # Write the mesh to file "cube.stl"
    cube.save('back_plate.stl')
    tstr=str(t) #thickness of the plate
    calt.t=t
    calt.a=a;
    calt.b=b;
    calt.t=t;
    calt.imax=imax;
    calt.jmax=jmax;
    calt.B=B;
    print(t)
    g_mesh();
    return(t)


def g_mesh():
    imax=calt.imax; Beta=calt.B; jmax=calt.jmax; L1=calt.b; L2=calt.t; #Step1
    xi=np.linspace(0,1,imax-1) #Step2
    xc=np.linspace(0,1,imax)
    Beta_p1=Beta+1;  Beta_m1=Beta-1;
    Beta_p1_div_m1=np.power((Beta_p1/Beta_m1),(2*xi-1));
    
    num=(Beta_p1*Beta_p1_div_m1)-Beta_m1; den=2*(1+Beta_p1_div_m1);
    x=L1*(np.divide(num,den)) #Step3
    print(x)
    y=x; 
    for i in range(1,imax-1):
        xc[i]=(x[i]+x[i-1])/2.0; #Step4
        xc[0]=x[0];   xc[imax-1]=x[imax-2];   
    yc=xc; #Step4
    Xc,Yc=np.meshgrid(xc,yc);  Zc=np.zeros([jmax,imax]);
    X,Y=np.meshgrid(x,y);      Z=np.zeros([jmax-1,imax-1]);
    Dx=np.linspace(0,1,imax-1)
    dx=np.linspace(0,1,imax)
    Dy=np.linspace(0,1,imax)
    dy=np.linspace(0,1,imax)
    for i in range(1,imax-1):
        Dx[i]=x[i]-x[i-1]
        Dy[i]=y[i]-y[i-1]
    for i in range(0,imax-1):
        dx[i]=xc[i+1]-xc[i]
        dy[i]=yc[i+1]-yc[i]
    #plot
        fig = plt.figure(1,figsize=(10,8))
    ax = plt.axes(projection='3d')
    ax.view_init(azim=0, elev=90)
    ax.plot_surface(X, Y, Z, edgecolor='black')
    plt.show()
    myName = "2Dmesh.png" 
    fig.savefig(myName)
    frame7=tk.Frame(root, bg='black', bd=5)#mid frame
    frame7.place(relx=0.5, rely=0.4, relwidth=0.75, relheight=0.5, anchor='n')
    label=tk.Label(frame7,bg='black')
    mesh_im=tk.PhotoImage(file="2Dmesh.png")
    label.place(relwidth=1,relheight=1)
    label['image']=mesh_im
    
    
    mesh.Dx=Dx
    mesh.Dy=Dy
    mesh.dx=dx
    mesh.dy=dy
    mesh.xc=xc
    mesh.yc=yc
    mesh.x=x
    #Re-entry for thermal
    
    #T_amb
    def lbt1(f):
        Status_label['text']='Ambient Temperature (K)'
        label1['bg']='red'
        return()
    
        
    entry1=tk.Entry(frame1, font=('Arial Black',8))
    entry1.place(relx=0.02,rely=0.09,relwidth= 0.5,relheight=0.8)
    label1['text']='T_amb'
    label1.bind("<Enter>",lbt1);

    
    #T_N
    def lbt2(f):
        Status_label['text']='Temperature of the Battery (K)'
        label2['bg']='red'
        return()
    
    label2['text']='T_Batt'
    entry2=tk.Entry(frame2, font=('Arial Black',8))
    entry2.place(relx=0.02,rely=0.09,relwidth= 0.5,relheight=0.8)
    label2.bind("<Enter>",lbt2);
    
    #T_wb
    def lbt5(f):
        Status_label['text']='Initial Temprature (K)'
        label5['bg']='red'
        return()
    label5['text']='T_int'
    entry5=tk.Entry(frame5, font=('Arial Black',8))
    entry5.place(relx=0.02,rely=0.09,relwidth= 0.5,relheight=0.8)
    label5.bind("<Enter>",lbt5);
    
    
    #Heat transfer coefficient
    def lbt3(f):
        Status_label['text']='Heat Transfer Coefficient (W/m^2*K)'
        label3['bg']='red'
        return()
    label3['text']='H.T.C'
    entry3=tk.Entry(frame3, font=('Arial Black',8))
    entry3.place(relx=0.02,rely=0.09,relwidth= 0.5,relheight=0.8)
    label3.bind("<Enter>",lbt3);
    
    #Density
    def lbt4(f):
        Status_label['text']='Density (Kg/m^3)'
        label4['bg']='red'
        return()
    label4['text']='Density'
    entry4=tk.Entry(frame4, font=('Arial Black',8))
    entry4.place(relx=0.02,rely=0.09,relwidth= 0.5,relheight=0.8)
    label4.bind("<Enter>",lbt4);
    
    #Cp
    def lbt_ms0(f):
        Status_label['text']='Specific Heat Capacity (J/Kg*K)'
        label_ms0['bg']='red'
        return()
    label_ms0['text']='Cp'
    entry_ms0=tk.Entry(frame_ms0, font=('Arial Black',8))
    entry_ms0.place(relx=0.02,rely=0.09,relwidth= 0.5,relheight=0.8)
    label_ms0.bind("<Enter>",lbt_ms0);
    
    
    #Coefficient of thermal conductivity
    def lbt_ms1(f):
        Status_label['text']='Thermal conductivity (W/m*K)'
        label_ms1['bg']='red'
        return()
    label_ms1['text']='k'
    entry_ms1=tk.Entry(frame_ms1, font=('Arial Black',8))
    entry_ms1.place(relx=0.02,rely=0.09,relwidth= 0.5,relheight=0.8)
    label_ms1.bind("<Enter>",lbt_ms1);
    
    #Heat Generation
    def lbt_ms2(f):
        Status_label['text']='Volumetric Heat Gen. (W/m^3*K)'
        label_ms2['bg']='red'
        return()
    label_ms2['text']='Heat Gen. Vol.'
    entry_ms2=tk.Entry(frame_ms2, font=('Arial Black',8))
    entry_ms2.place(relx=0.02,rely=0.09,relwidth= 0.5,relheight=0.8)
    label_ms2.bind("<Enter>",lbt_ms2);
    
    #Button
    def lbt_ms4(f):
        Status_label['text']='Please Wait!!'
        button_ms4['bg']='red'
        label['image']=""
        return()
    
    button_ms4=tk.Button(frame_ms4,text= "Thermal", font=('Arial Black',8), command=lambda: thermal_analysis(float(entry1.get()),float(entry2.get()),float(entry5.get()),float(entry3.get()),float(entry4.get()),float(entry_ms0.get()),float(entry_ms1.get()),float(entry_ms2.get())))
    button_ms4.place(relx=0.065,rely=0.05,relwidth=0.9,relheight=0.9)
    button_ms4.bind("<Enter>",lbt_ms4);
    g_mesh.mesh_im=mesh_im #protection from garbage value
    
    return()

def thermal_analysis(T_inf,T_N,T_wb,h,rho,cp,k,Q_vol_gen):
    def lbe_ms4(f):
        button_ms4['bg']='#e6e6e6'
        Status_label['text']='Temperature Contour'
        return()
    print(T_inf)
    print(T_N)
    print(T_wb)
    L1=1; L2=1; imax=calt.imax; jmax=calt.jmax;
    To=308;
    Q_vol_gen=0;
    epsilon_st=0.0001;
    Dt=1000;
    Q_gen=np.zeros(imax)
    Q_gen=np.multiply(Q_vol_gen,mesh.Dx); #Total Heat Generation
    aE=np.ones((imax,jmax))
    aN=np.ones((imax,jmax))
    aPo=np.ones((imax,jmax))
    aP=np.ones((imax,jmax))
    T=np.ones((imax,jmax))
    b=np.ones((imax,jmax))

    #Step2 compute geometrical perameters for the non uniform grid
    for j in range(1,jmax-1):
        for i in range(0,imax-1):
            aE[i,j]=np.round(k*(mesh.Dy[j]/mesh.dx[i]),decimals=20);
    for j in range(0,jmax-1):
        for i in range(1,imax-1):
            aN[i,j]=np.round(k*mesh.Dx[i]/mesh.dy[j],decimals=20);
    for j in range(1,jmax-1):
        for i in range(1,imax-1):
            aPo[i,j]=np.round(rho*cp*mesh.Dx[i]*mesh.Dy[j]/Dt,decimals=20);
            aP[i,j]=np.round(aPo[i,j]+aE[i,j]+aE[i-1,j]+aN[i,j]+aN[i,j-1],decimals=20);
            
            
            
    #Step3 IC and Dirichlet BCs
    for j in range(1,jmax-1): #Initial conditions
        for i in range(1,imax-1):
            T[i,j]=To; 
    for i in range(0,imax):
        T[0,i]=np.round(T_wb,decimals=20);#West
        T[i,0]=np.round(T_wb,decimals=20); #South
        T[imax-1,i]=np.round(T_wb,decimals=20); #East
    unsteadiness_nd=1;
    n=0;
    alpha=np.round(k/(rho*cp),decimals=20); 
    DTc=np.round(T_wb-T_inf,decimals=20);
    
    
    
    
    #=====Time Marching for Implicit LAEs starts===
    while (unsteadiness_nd>=epsilon_st):
        n=n+1;
        print(n,unsteadiness_nd)
        #Step4 Non-Dirichlet BCs and Consider the temperature as previous temperature
        for i in range(0,imax):
            T[i,jmax-1]=np.round(k*T[i,jmax-2]+(h*mesh.dy[jmax-2]*T_N)/(k+(h*mesh.dy[jmax-2])),decimals=20); #North
        T_old=np.round(T,decimals=20);
        for j in range(1,jmax-1):
            for i in range(1,imax-1):
                b[i,j]=np.round((aPo[i,j]*T_old[i,j]),decimals=20);
        #Step5 iterative solution (by GS method) at each time step
        
        epsilon=0.0001; N=0; Error=1;
        while (Error>=epsilon):
            N=N+1;
            #Non-Dirichlet BCs
            for i in range(0,imax):
                T[i,jmax-1]=np.around(((k*T[i,jmax-2])+(h*mesh.dy[jmax-2]*T_N))/(k+h*mesh.dy[jmax-2]),decimals=20); #North
            T_old_iter=np.around(T, decimals=20);
            for j in range(1,jmax-1):
                for i in range(1,imax-1):
                    T[i,j]=np.around(aE[i,j]*T[i+1,j]+ aE[i-1,j]*T[i-1,j] + aN[i,j]*T[i,j+1] + aN[i,j-1]*T[i,j-1] + b[i,j],decimals=20);
                    T[i,j]=np.around(T[i,j]/aP[i,j],decimals=10);
            Error=np.max((np.around(T,decimals=20)-np.around(T_old_iter,decimals=20)));
            print(Error)
    #Step6 Steady state convergence criterion
        unsteadiness=np.max((np.around((T-T_old),decimals=20)));
        unsteadiness_nd=np.around((unsteadiness*L1*L2/(alpha*DTc/Dt)),decimals=20);
    print(T) 
    colorinterpolation = 50
    colourMap = plt.cm.jet
    fig = plt.figure(1,figsize=(6.6,5))
    plt.title("Temperature Contours")
    plt.contourf(T, colorinterpolation, cmap=colourMap)
    plt.colorbar()
    plt.show()
    myNam= "Temp.png" 
    fig.savefig(myNam)
    frame10=tk.Frame(root, bg='black', bd=5)#mid frame
    frame10.place(relx=0.5, rely=0.4, relwidth=0.75, relheight=0.5, anchor='n')
    label10=tk.Label(frame10,bg='black')
    temp_im=tk.PhotoImage(file="Temp.png")
    label10.place(relwidth=1,relheight=1)
    label10['image']=temp_im
    thermal_analysis.temp_im=temp_im
    return()




root = tk.Tk() #create root
root.title('Back_plate')

canvas= tk.Canvas(root, height=1020, width=900)#window of the app
canvas.pack()
background_image=tk.PhotoImage(file='back.png')
background_label=tk.Label(root,image= background_image)
background_label.place(relwidth=1, relheight=1)

#info
st_frame=tk.Frame(root, bg='red', bd=2)
st_frame.place(relx=0.1, rely=0.95, relwidth=0.3, relheight=0.05, anchor='n')
Status_label=tk.Label(st_frame,text='',bd=1, relief='sunken',anchor='n');
Status_label.pack(fill='x',side='bottom',ipady=4)
    
#Structural Analysis        
def lb1(e):
    Status_label['text']='Mass of pack (Kg)'
    label1['bg']='red'
    return()
def lbe(e):
    label1['bg']='#e6e6e6'
    Status_label['text']=''
    return()
frame1=tk.Frame(root, bg='black', bd=4)#upper frame
frame1.place(relx=0.25, rely=0.1, relwidth=0.2, relheight=0.08, anchor='n')        
entry1=tk.Entry(frame1, font=('Arial Black',9))# to enter the data
entry1.place(relx=0.02,rely=0.09,relwidth= 0.5,relheight=0.8)               
label1=tk.Label(frame1,font=('Arial Black',10),anchor='c', justify='left',bd=4, text ='LOAD')
label1.place(relx= 0.5,rely=0.09,relwidth=0.5,relheight=0.8)
label1.bind("<Enter>",lb1);
label1.bind("<Leave>",lbe);



def lb2(e):
    Status_label['text']='Length (mm)'
    label2['bg']='red'
    return()
def lbe2(e):
    label2['bg']='#e6e6e6'
    Status_label['text']=''
    return()        
frame2=tk.Frame(root, bg='black', bd=4)#upper frame
frame2.place(relx=0.25, rely=0.2, relwidth=0.2, relheight=0.08, anchor='n')
entry2=tk.Entry(frame2, font=('Arial Black',9))# to enter the data
entry2.place(relx=0.02,rely=0.09,relwidth= 0.5,relheight=0.8)
label2=tk.Label(frame2,font=('Arial Black',10),anchor='c', justify='left',bd=4, text ='Length')
label2.place(relx= 0.5,rely=0.09,relwidth=0.5,relheight=0.8)   
label2.bind("<Enter>",lb2);
label2.bind("<Leave>",lbe2);


def lb3(e):
    Status_label['text']='Breadth (mm)'
    label3['bg']='red'
    return()
def lbe3(e):
    label3['bg']='#e6e6e6'
    Status_label['text']=''
    return()  
frame3=tk.Frame(root, bg='black', bd=4)#upper frame
frame3.place(relx=0.52, rely=0.1, relwidth=0.2, relheight=0.08, anchor='n')
entry3=tk.Entry(frame3, font=('Arial Black',9))# to enter the data
entry3.place(relx=0.02,rely=0.09,relwidth= 0.5,relheight=0.8)
label3=tk.Label(frame3,font=('Arial Black',9),anchor='c', justify='left',bd=4, text ='Breadth')
label3.place(relx= 0.5,rely=0.09,relwidth=0.5,relheight=0.8)
label3.bind("<Enter>",lb3);
label3.bind("<Leave>",lbe3);        
    
def lb4(e):
    Status_label['text']='Youngs Modulus (MPa)'
    label4['bg']='red'
    return()
def lbe4(e):
    label4['bg']='#e6e6e6'
    Status_label['text']=''
    return() 
frame4=tk.Frame(root, bg='black', bd=4)#upper frame
frame4.place(relx=0.52, rely=0.2, relwidth=0.2, relheight=0.08, anchor='n')
entry4=tk.Entry(frame4, font=('Arial Black',9))# to enter the data
entry4.place(relx=0.02,rely=0.09,relwidth= 0.5,relheight=0.8)
label4=tk.Label(frame4,font=('Arial Black',10),anchor='c', justify='left',bd=4, text ='Y mod')
label4.place(relx= 0.5,rely=0.09,relwidth=0.5,relheight=0.8)
label4.bind("<Enter>",lb4);
label4.bind("<Leave>",lbe4);

def lb5(e):
    Status_label['text']='permisible deflection (mm)'
    label5['bg']='red'
    return()
def lbe5(e):
    label5['bg']='#e6e6e6'
    Status_label['text']=''
    return() 
frame5=tk.Frame(root, bg='black', bd=4)#upper frame
frame5.place(relx=0.25, rely=0.3, relwidth=0.2, relheight=0.08, anchor='n')
entry5=tk.Entry(frame5, font=('Arial Black',9))# to enter the data
entry5.place(relx=0.02,rely=0.09,relwidth= 0.5,relheight=0.8)
label5=tk.Label(frame5,font=('Arial Black',8),anchor='c', justify='left',bd=4, text ='Def.')
label5.place(relx= 0.5,rely=0.09,relwidth=0.5,relheight=0.8)
label5.bind("<Enter>",lb5);
label5.bind("<Leave>",lbe5);      
     
#Structural Analysis ends    
   
#Lower Phi_Labs frame     
lowest_frame=tk.Frame(root, bg='red', bd=2)
lowest_frame.place(relx=0.95, rely=0.95, relwidth=0.1, relheight=0.05, anchor='n')
label_tg=tk.Label(lowest_frame,font=('Arial Black',7),anchor='se', justify='left',bd=4, text ='Phi_labs')
label_tg.place(relwidth=1,relheight=1)
        



#Mesh_generation

def lbms0(e):
    Status_label['text']='Beta: Hoffmann and Chiang,(2000)'
    label_ms0['bg']='red'
    return()
def lbems0(e):
    label_ms0['bg']='#e6e6e6'
    Status_label['text']=''
    return()
frame_ms0=tk.Frame(root, bg='black', bd=4)#upper frame
frame_ms0.place(relx=0.52, rely=0.3, relwidth=0.2, relheight=0.08, anchor='n')
entry_ms0=tk.Entry(frame_ms0, font=('Arial Black',9))# to enter the data
entry_ms0.place(relx=0.02,rely=0.09,relwidth= 0.5,relheight=0.8)
label_ms0=tk.Label(frame_ms0,font=('Arial Black',8),anchor='c', justify='left',bd=4, text ='Mesh(B)')
label_ms0.place(relx= 0.5,rely=0.09,relwidth=0.5,relheight=0.8)
label_ms0.bind("<Enter>",lbms0);
label_ms0.bind("<Leave>",lbems0);


def lbms1(e):
    Status_label['text']=''
    label_ms1['bg']='red'
    return()
def lbems1(e):
    label_ms1['bg']='#e6e6e6'
    Status_label['text']=''
    return()    
frame_ms1=tk.Frame(root, bg='black', bd=4)#upper frame
frame_ms1.place(relx=0.78, rely=0.1, relwidth=0.2, relheight=0.08, anchor='n')
entry_ms1=tk.Entry(frame_ms1, font=('Arial Black',9))# to enter the data
entry_ms1.place(relx=0.02,rely=0.09,relwidth= 0.5,relheight=0.8)
label_ms1=tk.Label(frame_ms1,font=('Arial Black',8),anchor='c', justify='left',bd=4, text ='Mesh(x)')
label_ms1.place(relx= 0.5,rely=0.09,relwidth=0.5,relheight=0.8)
label_ms1.bind("<Enter>",lbms1);
label_ms1.bind("<Leave>",lbems1);




def lbms2(e):
    Status_label['text']=''
    label_ms2['bg']='red'
    return()
def lbems2(e):
    label_ms2['bg']='#e6e6e6'
    Status_label['text']=''
    return()    
frame_ms2=tk.Frame(root, bg='black', bd=4)#upper frame
frame_ms2.place(relx=0.78, rely=0.2, relwidth=0.2, relheight=0.08, anchor='n')
entry_ms2=tk.Entry(frame_ms2, font=('Arial Black',8))# to enter the data
entry_ms2.place(relx=0.02,rely=0.09,relwidth= 0.5,relheight=0.8)
label_ms2=tk.Label(frame_ms2,font=('Arial Black',8),anchor='c', justify='left',bd=4, text ='Mesh(y)')
label_ms2.place(relx= 0.5,rely=0.09,relwidth=0.5,relheight=0.8)
label_ms2.bind("<Enter>",lbms2);
label_ms2.bind("<Leave>",lbems2);



def bms(e):
    Status_label['text']=''
    button_ms4['bg']='red'
    return()
def bems(e):
    button_ms4['bg']='#e6e6e6'
    Status_label['text']=''
    return()  
frame_ms4=tk.Frame(root, bg='black', bd=1)#upper frame
frame_ms4.place(relx=0.78, rely=0.3, relwidth=0.2, relheight=0.08, anchor='n')
button_ms4=tk.Button(frame_ms4,text= "Generate Mesh", font=('Arial Black',8), command=lambda: calt(float(entry1.get()),float(entry2.get()),float(entry3.get()),float(entry4.get()),float(entry5.get()),float(entry_ms0.get()),int(entry_ms1.get()),int(entry_ms2.get())))
button_ms4.place(relx=0.065,rely=0.05,relwidth=0.9,relheight=0.9)
button_ms4.bind("<Enter>",bms);
button_ms4.bind("<Leave>",bems);




root.mainloop()