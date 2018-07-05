# jx-ai-net

## train.py

Se eseguito procede all'addestramento di una rete utilizzando due file: `x.csv` e `y.csv`.

La locazione dei file `x.csv` e `y.csv` e altri parametri di controllo sono scritti in un file di configurazione (e.g. `./myjxai.conf`) che deve essere passato come unico parametro a `train.py`.  
Nel file di configurazione è anche possibile scegliere se effettuare un log su file e/o online  
Nel file presente nella root del progetto sono riportati i **valori di default**.

Il lancio dello script pertanto avviene così:

`python train.py ./myjxai.conf &`

L'esecuzione dello script porta alla produczione di un file di modello e.g.: `./myjxai.h5`.

## predict.py

Se eseguito procede alla classificazione di una rete utilizzando `x.csv` e produce un file di classificazione `y_hat.csv`.

Lo script prevede in ingresso lo stesso file di configurazione delle script precedente.  
Il lancio dello script pertanto avviene così:

`python predict.py ./myjxai.conf &`

### Test folder

Per il test sono presenti due file x e y in `./test_data`.  

- Il file `x.csv` rappresenta il vettore delle variabili in ingresso e contiene 900 samples di 8 variabili reali.

- Il file `y.csv` rappresenta il vettore delle variabili in uscita e contiene un egual numero di samplese rispetto a `x.csv` di 4 variabili reali.

La rete è una LSTM con uno strato denso in uscita.  
Gli ingressi e le uscite sono valori reali e normalizzati per ogni colonna tra 0 e 1 compresi.  
Nei file utilizzati per il test i valori hanno 16 decimali (in realtà il numero significativo di decimali sembra essere intorno agli 8).
