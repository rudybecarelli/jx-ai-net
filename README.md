# jx-ai-net

C'è un solo script: train.py.

Se eseguito, procede all'addestramento di una rete utilizzando due file: x.csv e y.csv.

La locazione dei file x.csv e y.cs e altri parametri di controllo sono scritti in un file di configurazione (e.g. `./myjxai.conf`) che deve essere passato come unico parametro a train.py.  
Nel file di configurazione è anche possibile scegliere se effettuare un log su file e/o su standard output.  
Nel file presente nel progetto sono riportati i **valori di default**.

Il lancio dello script pertanto avviene così:

`python train.py ./myjxai.conf`

Per il test sono presenti due file x e y in `./test_data`.  
Il file di configurazione è presente nella cartella di root.

- Il file x.csv rappresenta il vettore delle variabili in ingresso e contiene un elevato numero di samples (oltre 40.000) di 8 variabili reali.

- Il file y.csv rappresenta il vettore delle variabili in uscita e contiene un egual numero di samplese rispetto a x.csv di 4 variabili reali.

La rete è una LSTM con uno strato denso in uscita.  
Gli ingressi e le uscite sono valori reali e normalizzati per ogni colonna tra 0 e 1 compresi.  
Nei file utilizzati per il test i valori hanno 16 decimali (in realtà il numero significativo di decimali sembra essere intorno agli 8).

Una volta terminato l'addestramento, se l'opzione del file di configurazione `log_to_file` è selezionata, vengono salvati 2 grafici per successive analisi e un file di log che contiene due righe:

- Nella prima viene riportato il valore dell'RMSE sul validation set. Tale valore è quello che deve essere minimizzato

- Nella seconda viene riportata la predict (y_hat) che ha come input l'ultima riga del file x.csv

Se l'opzione `log_to_std_out` è selezionata le stesse informazioni vengono anche ritornate sull standard output.

Al termine di ogni addestramento viene anche salvato il modello.
