import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation,FuncAnimation
from tqdm import tqdm
from scipy import fft
import sys,os
import argparse
c=3*10**8                                                                                                                            #costante richiesta

def parse_arguments():                                                                                                               #Funzione Argparse
    
    parser = argparse.ArgumentParser(description="Pacchetti d'onda.",
                                     usage      ='python3 Progetto.py  --PacchettoN')
    parser.add_argument('-a', '--Pacchetto1',    action='store_true',                     help='Simulazione primo pacchetto')
    parser.add_argument('-b', '--Pacchetto2',    action='store_true',                     help='Simulazione secondo pacchetto')
    parser.add_argument('-c', '--Pacchetto3',    action='store_true',                     help='Simulazione terzo pacchetto')
    parser.add_argument('-d', '--Pacchetto4',    action='store_true',                     help='Simulazione quarto pacchetto')
    parser.add_argument('-e', '--Pacchetto5',    action='store_true',                     help='Simulazione quinto pacchetto')
    return  parser.parse_args()
def freq(N):                                                                                                                         #Funzione per trovare frequenze e ampiezze del pacchetto:
    frequenze=[]                                                                                                                     #Genera un numero,inserito dall'utente,  di frequenze e ampiezze con la 
    ampiezze=[]                                                                                                                      #distribuzione di probabilità data tramite il processo della cumulativa
    a=0
    b=5
    for i in range (N):
        u1=np.random.random()
        u2=np.random.random()
        if u1<=2/3:
            frequenza=np.sqrt(6*u1)#cumulativa con funzione a tratti
        else:
            frequenza=3-np.sqrt(3*(1-u1))
        Amax=3*np.sqrt(frequenza)
        ampiezza=np.sqrt(a**2+u2*(Amax**2-a**2))
        ampiezze.append(ampiezza)
        frequenze.append(frequenza)
    return np.array(frequenze),np.array(ampiezze)

def genera_pacchetto1(frequenze,ampiezze,spazi,tempo):                                                                               #Funzione per pacchetto senza dispersione:
    if(type(spazi)==int):                                                                                                            #utilizza le frequenze e le ampiezze generate sopra insime ai valori spazi
        pacchetto=np.zeros_like(tempo)                                                                                               #e tempo, spazi può essere sia un intero, sia un float, sia una lista a
    elif(type(spazi)==np.float64):                                                                                                   #seconda delle necessità, inoltre viene utilizzata la funzione di
        pacchetto=np.zeros_like(tempo)                                                                                               #dispersione che cambia a seconda del pacchetto.
    else:                                                                                                                            #Viene restituito un array di numpy che contiene le componenti del
        pacchetto=np.zeros_like(spazi)                                                                                               #pacchetto.
    
    k=2*np.pi*frequenze/c
    for i in range(len(frequenze)):
        fase=2*np.pi*frequenze[i]*tempo
        pacchetto+=ampiezze[i]*np.cos(k[i]*spazi-fase)
    return pacchetto

def genera_pacchetto2(frequenze,ampiezze,spazi,tempo):                                                                               #Funzione per pacchetto con prima dispersione
    if(type(spazi)==int):
        pacchetto=np.zeros_like(tempo)
    elif(type(spazi)==np.float64):
        pacchetto=np.zeros_like(tempo)
    else:
        pacchetto=np.zeros_like(spazi)
    k=(2*np.pi*frequenze)**2/c
    for i in range(len(frequenze)):
        fase=2*np.pi*frequenze[i]*tempo
        pacchetto+=ampiezze[i]*np.cos(k[i]*spazi-fase)
    return pacchetto

def genera_pacchetto3(frequenze,ampiezze,spazi,tempo):                                                                               #Funzione per pacchetto con seconda dispersione
    if(type(spazi)==int):
        pacchetto=np.zeros_like(tempo)
    elif(type(spazi)==np.float64):
        pacchetto=np.zeros_like(tempo)
    else:
        pacchetto=np.zeros_like(spazi)
    k=2*np.pi*frequenze/c**1/2
    for i in range(len(frequenze)):
        fase=2*np.pi*frequenze[i]*tempo
        pacchetto+=ampiezze[i]*np.cos(k[i]*spazi-fase)
    return pacchetto

def genera_pacchetto4(frequenze,ampiezze,spazi,tempo):                                                                               #Funzione per pacchetto con terza dispersione
    if(type(spazi)==int):
        pacchetto=np.zeros_like(tempo)
    elif(type(spazi)==np.float64):
        pacchetto=np.zeros_like(tempo)
    else:
        pacchetto=np.zeros_like(spazi)
    k=((2*np.pi*frequenze)**2/c)**1/3
    for i in range(len(frequenze)):
        fase=2*np.pi*frequenze[i]*tempo
        pacchetto+=ampiezze[i]*np.cos(k[i]*spazi-fase)
    return pacchetto

def genera_pacchetto5(frequenze,ampiezze,spazi,tempo,b):                                                                             #Funzione per pacchetto con quarta dispersione
    if(type(spazi)==int):                                                                                                           
        pacchetto=np.zeros_like(tempo)
    elif(type(spazi)==np.float64):
        pacchetto=np.zeros_like(tempo)
    else:
        pacchetto=np.zeros_like(spazi)
    k=(np.abs((2*np.pi*frequenze)**2-b))**1/2/c**1/2
    for i in range(len(frequenze)):
        fase=2*np.pi*frequenze[i]*tempo
        pacchetto+=ampiezze[i]*np.cos(k[i]*spazi-fase)
    return pacchetto

def main():                                                                                                                          #Funzione main
    args=parse_arguments()
    if args.Pacchetto1==True:                                                                                                        #PRIMO PACCHETTO
        N=int(input('Inserire il numero di elementi per il primo pacchetto (Range suggerito 1000-10000): '))
    

        frequenze,ampiezze=freq(N)                                                                                                   #Generazione e istogrammi di frequenze e ampiezze:
        plt.hist(frequenze,bins=50,label='Frequenze',density=True)                                                                   #Vengono generati con l'attributo density=true in modo da avere l'area 
        plt.title('Andamento Frequenze')                                                                                             #sottesa uguale a 1, inoltre vengono graficate anche le rette che seguono
        x0,y0=0,0                                                                                                                    #la distribuzione di probabilità attesa per le frequenze
        x1,y1=2,2*0.3
        plt.plot([x0,x1],[y0,y1],color='r',linestyle='--',label='Prima probabilità')
        x2,y2=3,0
        plt.plot([x1,x2],[y1,y2],color='y',linestyle='--',label='Seconda probabilità')
        plt.xlabel('Frequenze (Hz)')
        plt.ylabel('Probabilità')
        plt.legend()
        plt.show()
        plt.hist(ampiezze,bins=50,label='Ampiezze',density=True)
        plt.title('Andamento Ampiezze')
        plt.xlabel('Ampiezza (u.a.)')
        plt.ylabel('Probabilità')
        plt.legend()
        plt.show()

        spazio=np.linspace(-15,15,N)*10**8

        tempo=np.arange(-5,5,1/20) 


        pacchetto1=genera_pacchetto1(frequenze,ampiezze,spazio,0)                                                                    #Generazione pacchetto con distribuzione senza dispersione
        plt.plot(spazio,pacchetto1,color='#800080')
        plt.xlabel('Posizione') 
        plt.ylabel('Ampiezza')
        plt.title('Primo pacchetto')
        plt.show()

        fig,ax=plt.subplots()
        img1=[]

        for i in tqdm(tempo):                                                                                                       #Generazione animazione pacchetto senza dispersione
            frame= genera_pacchetto1(frequenze,ampiezze,spazio,i)
            im,=ax.plot(spazio,frame,animated=True,color='#800080')
            img1.append([im,])
        ani=ArtistAnimation(fig,img1,interval=50,blit=True,repeat_delay=1000)
        ax.set_title('Evoluzione primo pacchetto')
        ax.set_xlabel('Posizione')
        ax.set_ylabel('Ampiezza')
        plt.show()

        pacchetto1t=genera_pacchetto1(frequenze,ampiezze,0,tempo)

        ampiezza=fft.fft(pacchetto1t)                                                                                                #Trasformata di fourier del primo pacchetto
        frequenza=fft.fftfreq(pacchetto1t.size,d=tempo[0]-tempo[1])                                                                  #Viene plottata sia la potenza che le parti immaginarie e reali
        plt.plot(frequenza[len(ampiezza)//2:],np.abs(ampiezza[len(ampiezza)//2:])**2,color='#800080')
        plt.xlabel('Frequenza (Hz)')
        plt.ylabel('Potenza (u.a.)')
        plt.title('Spettro di potenza primo pacchetto')
        plt.show()
        plt.plot(frequenza[len(ampiezza)//2:],np.real(ampiezza[len(ampiezza)//2:]),color='#800080')
        plt.xlabel('Frequenza (Hz)')
        plt.ylabel('Ampiezza (u.a.)')
        plt.title('Parti reali primo pacchetto')
        plt.show()
        plt.plot(frequenza[len(ampiezza)//2:],np.imag(ampiezza[len(ampiezza)//2:]),color='#800080')
        plt.xlabel('Frequenza (Hz)')
        plt.ylabel('Ampiezza (u.a.)')
        plt.title('Parti immaginarie primo pacchetto')
        plt.show()


        fig,ax=plt.subplots()                                                                                                       #Evoluzione componenti della trasformata primo pacchetto
        anispettro=[]


        for i in tqdm(spazio):
            frame= genera_pacchetto1(frequenze,ampiezze,i,tempo)
            ampiezza = fft.fft(frame)
            frequenza = fft.fftfreq(frame.size, d=tempo[1] - tempo[0] )
            im,=ax.plot(frequenza[:len(ampiezza)//2],np.abs(ampiezza[:len(ampiezza)//2])**2,animated=True,color='#800080')
            anispettro.append([im,])
        ani=ArtistAnimation(fig,anispettro,interval=50,blit=True,repeat_delay=1000)
        ax.set_title('Evoluzione spettro di potenza primo pacchetto')
        ax.set_xlabel('Frequenza(Hz)')
        ax.set_ylabel('Ampiezza (u.a.)')
        plt.show()

        fig,ax=plt.subplots()
        anireal=[]


        for i in tqdm(spazio):
            frame= genera_pacchetto1(frequenze,ampiezze,i,tempo)
            ampiezza = fft.fft(frame)
            frequenza = fft.fftfreq(frame.size, d=(tempo[1] - tempo[0]))
            im,=ax.plot(frequenza[:len(ampiezza)//2],np.real(ampiezza[:len(ampiezza)//2]),animated=True,color='#800080')
            anireal.append([im,])
        ani=ArtistAnimation(fig,anireal,interval=50,blit=True,repeat_delay=1000)
        ax.set_title('Evoluzione parti reali primo pacchetto')
        ax.set_xlabel('Frequenza (Hz)')
        ax.set_ylabel('Ampiezza (u.a.)')
        plt.show()

        fig,ax=plt.subplots()
        aniimag=[]


        for i in tqdm(spazio):
            frame= genera_pacchetto1(frequenze,ampiezze,i,tempo)
            ampiezza = fft.fft(frame)
            frequenza = fft.fftfreq(frame.size, d=(tempo[1] - tempo[0]))
            im,=ax.plot(frequenza[:len(ampiezza)//2],np.imag(ampiezza[:len(ampiezza)//2]),animated=True,color='#800080')
            aniimag.append([im,])
        ani=ArtistAnimation(fig,aniimag,interval=50,blit=True,repeat_delay=1000)
        ax.set_title('Evoluzione parti immaginarie primo pacchetto')
        ax.set_xlabel('Frequenza (Hz)')
        ax.set_ylabel('Ampiezza (u.a.)')
        plt.show()
        
    if args.Pacchetto2==True:                                                                                                                #SECONDO PACCHETTO
        N=int(input('Inserire il numero di elementi per il secondo pacchetto (Range suggerito 1000-10000): '))
    

        frequenze,ampiezze=freq(N)                                                                                                           #Generazione e istogrammi di frequenze e ampiezze
        plt.hist(frequenze,bins=50,label='Frequenze',density=True)
        plt.title('Andamento Frequenze')
        x0,y0=0,0
        x1,y1=2,2*0.3
        plt.plot([x0,x1],[y0,y1],color='r',linestyle='--',label='Prima probabilità')
        x2,y2=3,0
        plt.plot([x1,x2],[y1,y2],color='y',linestyle='--',label='Seconda probabilità')
        plt.xlabel('Frequenze (Hz)')
        plt.ylabel('Probabilità')
        plt.legend()
        plt.show()
        plt.hist(ampiezze,bins=50,label='Ampiezze',density=True)
        plt.title('Andamento Ampiezze')
        plt.xlabel('Ampiezza (u.a.)')
        plt.ylabel('Probabilità')
        plt.legend()
        plt.show()

        spazio=np.linspace(-15,15,N)*10**8
        tempo=np.arange(-5,5,1/20) 

        pacchetto2=genera_pacchetto2(frequenze,ampiezze,spazio,0)                                                                           #Generazione pacchetto con prima dispersione
        plt.plot(spazio,pacchetto2,color='#FFA500')
        plt.xlabel('Posizione')
        plt.ylabel('Ampiezza')
        plt.title('Secondo pacchetto')
        plt.show()

        fig,ax=plt.subplots()
        img2=[]

        for i in tqdm(tempo):                                                                                                               #Evoluzione pacchetto prima dispersione
            frame= genera_pacchetto2(frequenze,ampiezze,spazio,i)
            im,=ax.plot(spazio,frame,animated=True,color='#FFA500')
            img2.append([im,])
        ani=ArtistAnimation(fig,img2,interval=50,blit=True,repeat_delay=1000)
        ax.set_title('Evoluzione secondo pacchetto')
        ax.set_xlabel('Posizione')
        ax.set_ylabel('Ampiezza')
        plt.show()

        pacchetto2t=genera_pacchetto2(frequenze,ampiezze,0,tempo)

        ampiezza=fft.fft(pacchetto2t)                                                                                                       #Trasformata di Fourier 
        frequenza=fft.fftfreq(pacchetto2t.size,d=tempo[0]-tempo[1])
        plt.plot(frequenza[len(ampiezza)//2:],np.abs(ampiezza[len(ampiezza)//2:])**2,color='#FFA500')
        plt.xlabel('Frequenza (Hz)')
        plt.ylabel('Potenza (u.a.)')
        plt.title('Spettro di potenza secondo pacchetto')
        plt.show()
        plt.plot(frequenza[len(ampiezza)//2:],np.real(ampiezza[len(ampiezza)//2:]),color='#FFA500')
        plt.xlabel('Frequenza (Hz)')
        plt.ylabel('Ampiezza (u.a.)')
        plt.title('Parti reali secondo pacchetto')
        plt.show()
        plt.plot(frequenza[len(ampiezza)//2:],np.imag(ampiezza[len(ampiezza)//2:]),color='#FFA500')
        plt.xlabel('Frequenza (Hz)')
        plt.ylabel('Ampiezza (u.a.)')
        plt.title('Parti immaginarie secondo pacchetto')
        plt.show()



        fig,ax=plt.subplots()
        anispettro=[]


        for i in tqdm(spazio):                                                                                                            #Evoluzione componenti della trasformata primo pacchetto
            frame= genera_pacchetto2(frequenze,ampiezze,i,tempo)
            ampiezza = fft.fft(frame)
            frequenza = fft.fftfreq(frame.size, d=tempo[1] - tempo[0] )
            im,=ax.plot(frequenza[:len(ampiezza)//2],np.abs(ampiezza[:len(ampiezza)//2])**2,animated=True,color='#FFA500')
            anispettro.append([im,])
        ani=ArtistAnimation(fig,anispettro,interval=50,blit=True,repeat_delay=1000)
        ax.set_title('Evoluzione spettro di potenza secondo pacchetto')
        ax.set_xlabel('Frequenza (Hz)')
        ax.set_ylabel('Potenza (u.a.)')
        plt.show()

        fig,ax=plt.subplots()
        anireal=[]


        for i in tqdm(spazio):
            frame= genera_pacchetto2(frequenze,ampiezze,i,tempo)
            ampiezza = fft.fft(frame)
            frequenza = fft.fftfreq(frame.size, d=(tempo[1] - tempo[0]))
            im,=ax.plot(frequenza[:len(ampiezza)//2],np.real(ampiezza[:len(ampiezza)//2]),animated=True,color='#FFA500')
            anireal.append([im,])
        ani=ArtistAnimation(fig,anireal,interval=50,blit=True,repeat_delay=1000)
        ax.set_title('Evoluzione parti reali secondo pacchetto')
        ax.set_xlabel('Frequenza (Hz)')
        ax.set_ylabel('Ampiezza (u.a.) ')
        plt.show()

        fig,ax=plt.subplots()
        aniimag=[]

        for i in tqdm(spazio):
            frame= genera_pacchetto2(frequenze,ampiezze,i,tempo)
            ampiezza = fft.fft(frame)
            frequenza = fft.fftfreq(frame.size, d=(tempo[1] - tempo[0]))
            im,=ax.plot(frequenza[:len(ampiezza)//2],np.imag(ampiezza[:len(ampiezza)//2]),animated=True,color='#FFA500')
            aniimag.append([im,])
        ani=ArtistAnimation(fig,aniimag,interval=50,blit=True,repeat_delay=1000)
        ax.set_title('Evoluzione parti immaginarie secondo pacchetto')
        ax.set_xlabel('Frequenza (Hz)')
        ax.set_ylabel('Ampiezza (u.a.)')
        plt.show()
    if args.Pacchetto3==True:                                                                                                                   #TERZO PACCHETTO
        N=int(input('Inserire il numero di elementi per il terzo pacchetto (Range suggerito 1000-10000): '))
    

        frequenze,ampiezze=freq(N)                                                                                                               #Generazione e istogrammi di frequenze e ampiezze
        plt.hist(frequenze,bins=50,label='Frequenze',density=True)
        plt.title('Andamento Frequenze')
        x0,y0=0,0
        x1,y1=2,2*0.3
        plt.plot([x0,x1],[y0,y1],color='r',linestyle='--',label='Prima probabilità')
        x2,y2=3,0
        plt.plot([x1,x2],[y1,y2],color='y',linestyle='--',label='Seconda probabilità')
        plt.xlabel('Frequenze (Hz)')
        plt.ylabel('Probabilità')
        plt.legend()
        plt.show()
        plt.hist(ampiezze,bins=50,label='Ampiezze',density=True)
        plt.title('Andamento Ampiezze')
        plt.xlabel('Ampiezza (u.a.)')
        plt.ylabel('Probabilità')
        plt.legend()
        plt.show()

        spazio=np.linspace(-15,15,N)*10**8
        tempo=np.arange(-5,5,1/20) 

        pacchetto3=genera_pacchetto3(frequenze,ampiezze,spazio,0)                                                                               #Generazione pacchetto con seconda dispersione

        plt.plot(spazio,pacchetto3,color='#00C957')
        plt.xlabel('Posizione')
        plt.ylabel('Ampiezza')
        plt.title('Terzo pacchetto')
        plt.show()
        fig,ax=plt.subplots()
        img3=[]

        for i in tqdm(tempo):                                                                                                                   #Animazione pacchetto
            frame= genera_pacchetto3(frequenze,ampiezze,spazio,i)
            im,=ax.plot(spazio,frame,animated=True,color='#00C957')
            img3.append([im,])
        ani=ArtistAnimation(fig,img3,interval=50,blit=True,repeat_delay=1000)
        ax.set_title('Evoluzione terzo pacchetto')
        ax.set_xlabel('Posizione')
        ax.set_ylabel('Ampiezza')
        plt.show()

        pacchetto3t=genera_pacchetto3(frequenze,ampiezze,0,tempo)

        ampiezza=fft.fft(pacchetto3t)                                                                                                           #Trasformata di Fourier
        frequenza=fft.fftfreq(pacchetto3t.size,d=tempo[0]-tempo[1])


        plt.plot(frequenza[len(ampiezza)//2:],np.abs(ampiezza[len(ampiezza)//2:])**2,color='#00C957')
        plt.xlabel('Frequenza (Hz)')
        plt.ylabel('Potenza (u.a.)')
        plt.title('Spettro di potenza terzo pacchetto')
        plt.show()
        plt.plot(frequenza[len(ampiezza)//2:],np.real(ampiezza[len(ampiezza)//2:]),color='#00C957')
        plt.xlabel('Frequenza (Hz)')
        plt.ylabel('Ampiezza (u.a.)')
        plt.title('Parti reali terzo pacchetto')
        plt.show()
        plt.plot(frequenza[len(ampiezza)//2:],np.imag(ampiezza[len(ampiezza)//2:]),color='#00C957')
        plt.xlabel('Frequenza (Hz)')
        plt.ylabel('Ampiezza (u.a.)')
        plt.title('Parti immaginarie terzo pacchetto')
        plt.show()




        fig,ax=plt.subplots()
        anispettro=[]

        
        for i in tqdm(spazio):                                                                                                                 #Evoluzione componenti della trasformata
            frame= genera_pacchetto3(frequenze,ampiezze,i,tempo)
            ampiezza = fft.fft(frame)
            frequenza = fft.fftfreq(frame.size, d=tempo[1] - tempo[0] )
            im,=ax.plot(frequenza[:len(ampiezza)//2],np.abs(ampiezza[:len(ampiezza)//2])**2,animated=True,color='#00C957')
            anispettro.append([im,])
        ani=ArtistAnimation(fig,anispettro,interval=50,blit=True,repeat_delay=1000)
        ax.set_title('Evoluzione spettro di potenza terzo pacchetto')
        ax.set_xlabel('Frequenza (Hz)')
        ax.set_ylabel('Potenza (u.a.)')
        plt.show()

        fig,ax=plt.subplots()
        anireal=[]


        for i in tqdm(spazio):
            frame= genera_pacchetto3(frequenze,ampiezze,i,tempo)
            ampiezza = fft.fft(frame)
            frequenza = fft.fftfreq(frame.size, d=(tempo[1] - tempo[0]))
            im,=ax.plot(frequenza[:len(ampiezza)//2],np.real(ampiezza[:len(ampiezza)//2]),animated=True,color='#00C957')
            anireal.append([im,])
        ani=ArtistAnimation(fig,anireal,interval=50,blit=True,repeat_delay=1000)
        ax.set_title('Evoluzione parti reali terzo pacchetto')
        ax.set_xlabel('Frequenza (Hz)')
        ax.set_ylabel('Ampiezza (u.a.)')
        plt.show()

        fig,ax=plt.subplots()
        aniimag=[]

        for i in tqdm(spazio):
            frame= genera_pacchetto3(frequenze,ampiezze,i,tempo)
            ampiezza = fft.fft(frame)
            frequenza = fft.fftfreq(frame.size, d=(tempo[1] - tempo[0]))
            im,=ax.plot(frequenza[:len(ampiezza)//2],np.imag(ampiezza[:len(ampiezza)//2]),animated=True,color='#00C957')
            aniimag.append([im,])
        ani=ArtistAnimation(fig,aniimag,interval=50,blit=True,repeat_delay=1000)
        ax.set_title('Evoluzione parti immaginarie terzo pacchetto')
        ax.set_xlabel('Frequenza (Hz)')
        ax.set_ylabel('Ampiezza (u.a.)')
        plt.show()  
    if args.Pacchetto4==True:                                                                                                                  #QUARTO PACCHETTO
        N=int(input('Inserire il numero di elementi per il quarto pacchetto (Range suggerito 1000-10000): '))
    

        frequenze,ampiezze=freq(N)                                                                                                             #Generazione e istogrammi di frequenze e ampiezze
        plt.hist(frequenze,bins=50,label='Frequenze',density=True)
        plt.title('Andamento Frequenze')
        x0,y0=0,0
        x1,y1=2,2*0.3
        plt.plot([x0,x1],[y0,y1],color='r',linestyle='--',label='Prima probabilità')
        x2,y2=3,0
        plt.plot([x1,x2],[y1,y2],color='y',linestyle='--',label='Seconda probabilità')
        plt.xlabel('Frequenze (Hz)')
        plt.ylabel('Probabilità')
        plt.legend()
        plt.show()
        plt.hist(ampiezze,bins=50,label='Ampiezze',density=True)
        plt.title('Andamento Ampiezze')
        plt.xlabel('Ampiezza (u.a.)')
        plt.ylabel('Probabilità')
        plt.legend()
        plt.show()

        spazio=np.linspace(-15,15,N)*10**8
        tempo=np.arange(-5,5,1/20) 

        pacchetto4=genera_pacchetto4(frequenze,ampiezze,spazio,0)                                                                              #Generazione pacchetto con terza dispersione

        plt.plot(spazio,pacchetto4,color='#000080')
        plt.xlabel('Posizione')
        plt.ylabel('Ampiezza')
        plt.title('Quarto pacchetto')
        plt.show()
        
        fig,ax=plt.subplots()
        img4=[]

        for i in tqdm(tempo):                                                                                                                  #Animazione quarto pacchetto
            frame= genera_pacchetto4(frequenze,ampiezze,spazio,i)
            im,=ax.plot(spazio,frame,animated=True,color='#000080')
            img4.append([im,])
        ani=ArtistAnimation(fig,img4,interval=50,blit=True,repeat_delay=1000)
        ax.set_title('Evoluzione quarto pacchetto')
        ax.set_xlabel('Posizione')
        ax.set_ylabel('Ampiezza')
        plt.show()

        pacchetto4t=genera_pacchetto4(frequenze,ampiezze,0,tempo)
        ampiezza=fft.fft(pacchetto4t)                                                                                                        #Trasformata di Fourier
        frequenza=fft.fftfreq(pacchetto4t.size,d=tempo[0]-tempo[1])

        plt.plot(frequenza[len(ampiezza)//2:],np.abs(ampiezza[len(ampiezza)//2:])**2,color='#000080')
        plt.xlabel('Frequenza (Hz)')
        plt.ylabel('Potenza (u.a.)')
        plt.title('Spettro di potenza Quarto pacchetto')
        plt.show()
        plt.plot(frequenza[len(ampiezza)//2:],np.real(ampiezza[len(ampiezza)//2:]),color='#000080')
        plt.xlabel('Frequenza (Hz)')
        plt.ylabel('Ampiezza (u.a.)')
        plt.title('Parti reali quarto pacchetto')
        plt.show()
        plt.plot(frequenza[len(ampiezza)//2:],np.imag(ampiezza[len(ampiezza)//2:]),color='#000080')
        plt.xlabel('Frequenza (Hz)')
        plt.ylabel('Ampiezza (u.a.)')
        plt.title('Parti immaginarie quarto pacchetto')
        plt.show()



        fig,ax=plt.subplots()
        anispettro=[]


        for i in tqdm(spazio):                                                                                                                #Evoluzione componenti della trasformata
            frame= genera_pacchetto4(frequenze,ampiezze,i,tempo)
            ampiezza = fft.fft(frame)
            frequenza = fft.fftfreq(frame.size, d=tempo[1] - tempo[0] )
            im,=ax.plot(frequenza[:len(ampiezza)//2],np.abs(ampiezza[:len(ampiezza)//2])**2,animated=True,color='#000080')
            anispettro.append([im,])
        ani=ArtistAnimation(fig,anispettro,interval=50,blit=True,repeat_delay=1000)
        ax.set_title('Evoluzione spettro di potenza quarto pacchetto')
        ax.set_xlabel('Frequenza (Hz)')
        ax.set_ylabel('Ampiezza (u.a.)')
        plt.show()

        fig,ax=plt.subplots()
        anireal=[]


        for i in tqdm(spazio):
            frame= genera_pacchetto4(frequenze,ampiezze,i,tempo)
            ampiezza = fft.fft(frame)
            frequenza = fft.fftfreq(frame.size, d=(tempo[1] - tempo[0]))
            im,=ax.plot(frequenza[:len(ampiezza)//2],np.real(ampiezza[:len(ampiezza)//2]),animated=True,color='#000080')
            anireal.append([im,])
        ani=ArtistAnimation(fig,anireal,interval=50,blit=True,repeat_delay=1000)
        ax.set_title('Evoluzione parti reali quarto pacchetto')
        ax.set_xlabel('Frequenza (Hz)')
        ax.set_ylabel('Ampiezza (u.a.)')
        plt.show()

        fig,ax=plt.subplots()
        aniimag=[]

        for i in tqdm(spazio):
            frame= genera_pacchetto4(frequenze,ampiezze,i,tempo)
            ampiezza = fft.fft(frame)
            frequenza = fft.fftfreq(frame.size, d=(tempo[1] - tempo[0]))
            im,=ax.plot(frequenza[:len(ampiezza)//2],np.imag(ampiezza[:len(ampiezza)//2]),animated=True,color='#000080')
            aniimag.append([im,])
        ani=ArtistAnimation(fig,aniimag,interval=50,blit=True,repeat_delay=1000)
        ax.set_title('Evoluzione parti immaginarie quarto pacchetto')
        ax.set_xlabel('Frequenza (Hz)')
        ax.set_ylabel('Ampiezza (u.a.)')
        plt.show()
    if args.Pacchetto5==True:                                                                                                                           #QUINTO PACCHETTO
        N=int(input('Inserire il numero di elementi per il quinto pacchetto (Range suggerito 1000-10000): '))
        b=int(input('Inserire la costante richiesta per la relazione di dispersione: '))

        frequenze,ampiezze=freq(N)                                                                                                                      #Generazione e istogrammi di frequenze e ampiezze
        plt.hist(frequenze,bins=50,label='Frequenze',density=True)
        plt.title('Andamento Frequenze')
        x0,y0=0,0
        x1,y1=2,2*0.3
        plt.plot([x0,x1],[y0,y1],color='r',linestyle='--',label='Prima probabilità')
        x2,y2=3,0
        plt.plot([x1,x2],[y1,y2],color='y',linestyle='--',label='Seconda probabilità')
        plt.xlabel('Frequenze (Hz)')
        plt.ylabel('Probabilità')
        plt.legend()
        plt.show()
        plt.hist(ampiezze,bins=50,label='Ampiezze',density=True)
        plt.title('Andamento Ampiezze')
        plt.xlabel('Ampiezza (u.a.)')
        plt.ylabel('Probabilità')
        plt.legend()
        plt.show()

        spazio=np.linspace(-15,15,N)*10**8
        tempo=np.arange(-5,5,1/20) 

        pacchetto5=genera_pacchetto5(frequenze,ampiezze,spazio,0,b)                                                                                   #Generazione pacchetto con quarta dispersione

        plt.plot(spazio,pacchetto5,color='#FF4040')
        plt.xlabel('Posizione')
        plt.ylabel('Ampiezza')
        plt.title('Quinto pacchetto')
        plt.show()

        fig,ax=plt.subplots()
        img5=[]

        for i in tqdm(tempo):
            frame= genera_pacchetto5(frequenze,ampiezze,spazio,i,b)
            im,=ax.plot(spazio,frame,animated=True,color='#FF4040')
            img5.append([im,])
        ani=ArtistAnimation(fig,img5,interval=50,blit=True,repeat_delay=1000)
        ax.set_title('Evoluzione quinto pacchetto')
        ax.set_xlabel('Posizione')
        ax.set_ylabel('Ampiezza')
        plt.show()

        pacchetto5t=genera_pacchetto5(frequenze,ampiezze,0,tempo,b)

        ampiezza=fft.fft(pacchetto5t)                                                                                                               #Trasformata di Jean-Baptiste Joseph Fourier
        frequenza=fft.fftfreq(pacchetto5t.size,d=tempo[0]-tempo[1])
        plt.plot(frequenza[len(ampiezza)//2:],np.abs(ampiezza[len(ampiezza)//2:])**2,color='#000080')
        plt.xlabel('Frequenza (Hz)')
        plt.ylabel('Potenza (u.a.)')
        plt.title('Spettro di potenza Quinto pacchetto')
        plt.show()
        plt.plot(frequenza[len(ampiezza)//2:],np.real(ampiezza[len(ampiezza)//2:]),color='#000080')
        plt.xlabel('Frequenza (Hz)')
        plt.ylabel('Ampiezza (u.a.)')
        plt.title('Parti reali quinto pacchetto')
        plt.show()
        plt.plot(frequenza[len(ampiezza)//2:],np.imag(ampiezza[len(ampiezza)//2:]),color='#000080')
        plt.xlabel('Frequenza (Hz)')
        plt.ylabel('Ampiezza (u.a.)')
        plt.title('Parti immaginarie quinto pacchetto')
        plt.show()



        fig,ax=plt.subplots()
        anispettro=[]


        for i in tqdm(spazio):                                                                                                                    #Evoluzione componenti della trasformata
            frame= genera_pacchetto5(frequenze,ampiezze,i,tempo,b)
            ampiezza = fft.fft(frame)
            frequenza = fft.fftfreq(frame.size, d=tempo[1] - tempo[0] )
            im,=ax.plot(frequenza[:len(ampiezza)//2],np.abs(ampiezza[:len(ampiezza)//2])**2,animated=True,color='#000080')
            anispettro.append([im,])
        ani=ArtistAnimation(fig,anispettro,interval=50,blit=True,repeat_delay=1000)
        ax.set_title('Evoluzione spettro di potenza quinto pacchetto')
        ax.set_xlabel('Frequenza (Hz)')
        ax.set_ylabel('Potenza (u.a.)')
        plt.show()

        fig,ax=plt.subplots()
        anireal=[]


        for i in tqdm(spazio):
            frame= genera_pacchetto5(frequenze,ampiezze,i,tempo,b)
            ampiezza = fft.fft(frame)
            frequenza = fft.fftfreq(frame.size, d=(tempo[1] - tempo[0]))
            im,=ax.plot(frequenza[:len(ampiezza)//2],np.real(ampiezza[:len(ampiezza)//2]),animated=True,color='#000080')
            anireal.append([im,])
        ani=ArtistAnimation(fig,anireal,interval=50,blit=True,repeat_delay=1000)
        ax.set_title('Evoluzione parti reali quinto pacchetto')
        ax.set_xlabel('Frequenza (Hz)')
        ax.set_ylabel('Ampiezza (u.a.)')
        plt.show()

        fig,ax=plt.subplots()
        aniimag=[]

        for i in tqdm(spazio):
            frame= genera_pacchetto5(frequenze,ampiezze,i,tempo,b)
            ampiezza = fft.fft(frame)
            frequenza = fft.fftfreq(frame.size, d=(tempo[1] - tempo[0]))
            im,=ax.plot(frequenza[:len(ampiezza)//2],np.imag(ampiezza[:len(ampiezza)//2]),animated=True,color='#000080')
            aniimag.append([im,])
        ani=ArtistAnimation(fig,aniimag,interval=50,blit=True,repeat_delay=1000)
        ax.set_title('Evoluzione parti immaginarie quinto pacchetto')
        ax.set_xlabel('Frequenza (Hz)')
        ax.set_ylabel('Ampiezza (u.a.)')
        plt.show()

if __name__=="__main__":
    main()
