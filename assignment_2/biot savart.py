import numpy as np
import matplotlib.pyplot as plt


def biot_savart(X1,X2,Xp,gamma): #TODO: Add CORE like in answer_from_the_github_page.js
    R1=np.sqrt((Xp[0]-X1[0])**2+(Xp[1]-X1[1])**2+(Xp[2]-X1[2])**2)
    R2=np.sqrt((Xp[0]-X2[0])**2+(Xp[1]-X2[1])**2+(Xp[2]-X2[2])**2)
    R12x=(Xp[1]-X1[1])*(Xp[2]-X2[2])-(Xp[2]-X1[2])*(Xp[1]-X2[1])
    R12y=-(Xp[0]-X1[0])*(Xp[2]-X2[2])+(Xp[2]-X1[2])*(Xp[0]-X2[0])
    R12z=(Xp[0]-X1[0])*(Xp[1]-X2[1])-(Xp[1]-X1[1])*(Xp[0]-X2[0])
    R12sqrt=R12x**2+R12y**2+R12z**2
    R01=(X2[0]-X1[0])*(Xp[0]-X1[0])+(X2[1]-X1[1])*(Xp[1]-X1[1])+(X2[2]-X1[2])*(Xp[2]-X1[2])
    R02=(X2[0]-X1[0])*(Xp[0]-X2[0])+(X2[1]-X1[1])*(Xp[1]-X2[1])+(X2[2]-X1[2])*(Xp[2]-X2[2])
    K=gamma/(4*np.pi*R12sqrt)*(R01/R1-R02/R2)
    U_ind=[float(K*(R12x)),float(K*(R12y)),float(K*(R12z))]
    return U_ind


print(biot_savart([1,0,0.3],[0,0,1],[0,0.5,0.5],15))

    
def calc_ind_filiment(Xp,r,tend=5):
    # tend=50
    dt=0.1
    # r=4
    omega=1
    tlst=np.arange(0,tend,dt)
    Uwake=10
    xarr=tlst*Uwake
    yarr=r*np.sin(omega*tlst)
    zarr=r*np.cos(omega*tlst)

    fig=plt.figure()
    fig2=plt.figure()
    ax=fig.subplots(2,2)
    ax2=fig2.add_subplot(projection='3d')
    uind=[]
    for i in range(len(tlst)-1):
        uind.append(biot_savart([xarr[i],yarr[i],zarr[i]],[xarr[i+1],yarr[i+1],zarr[i+1]],Xp,1)[0])
        ax[0,0].plot([xarr[i],xarr[i+1]],[yarr[i],yarr[i+1]])
        ax[0,1].plot([xarr[i],xarr[i+1]],[zarr[i],zarr[i+1]])
        ax[1,0].plot([yarr[i],yarr[i+1]],[zarr[i],zarr[i+1]])
        ax2.plot([xarr[i],xarr[i+1]],[yarr[i],yarr[i+1]],[zarr[i],zarr[i+1]],color='tab:blue')
        # ax2.plot([xarr2[i],xarr2[i+1]],[yarr2[i],yarr2[i+1]],[zarr2[i],zarr2[i+1]],color='tab:blue')
    ax[1,1].plot(uind)
    ax[1,1].grid()
    # surf=ax.plot_surface(mesh.X_nodes,mesh.Y_nodes,IC,cmap=cm.coolwarm)
    ax2.scatter(Xp[0],Xp[1],Xp[2])
    print(np.sum(uind))
    plt.show()

calc_ind_filiment([0,0,-5],4)




