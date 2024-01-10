# Progetto-MCF-2023
Repository contenente il progetto per l'esame di Metodi Computazionali per la Fisica




Il file è un file.py, contenente la simulazione richiesta.

Per avviare il programma digitare il nome del file seguito dal numero del pacchetto desiderato, in questa modalità: python3 Progetto.py -PacchettoN, sostituendo N con un numero da 1 a 5
(tutto questo è spiegato eseguendo anche python3 Progetto.py --help)

Eseguendo uno qualsiasi dei pacchetti, verrà richiesto il numero di elementi costituenti il pacchetto stesso, il range suggerito è tra 1000 e 10000, questo perchè per numeri inferiori la distribuzione delle frequenze, essendo generata casualmente, potrebbe non seguire l'andamento desiderato; mentre per numeri superiori il tempo di compilazione potrebbe essere troppo lungo.
Comunque ogni processo 'lungo' è accompagnato da una barra di avanzamento per rendersi conto del tempo di compilazione stesso.

Il primo grafico che viene visualizzato è l'istogramma, normalizzato, delle frequenze, graficato insieme agli andamenti previsti;
il secondo grafico è l'istogramma delle ampiezze corrispondenti, anch'esso normalizzato.

Quindi verrà mostrato il grafico del pacchetto generato, successivamente si avrà la prima barra di avanzamento per generare l'animazione del pacchetto stesso.

In seguito verrà mostrato anche lo spettro di potenza, le parti reali e immaginarie della trasformata di Fourier del segnale, con le relative animazioni (anche qui con le loro barre di avanzamento).

La differenza tra i pacchetti sta nella relazione di dispersione degli stessi, in più sono state fissate delle costanti:
c è stata messa uguale alla velocità della luce (3*10^8 m/s), a è stata fissata a 0, mentre l'ultima costante è scelta dall'utente nella generazione del pacchetto 5.
