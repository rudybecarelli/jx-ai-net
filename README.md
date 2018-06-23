# jx-ai-net

**Questa è la versione demo!**
---
C'è un solo eseguibile: train.py.

Se eseguito, procede all'addestramento di una rete utilizzando i file x.csv e y*.csv.
Il file x.csv contiene un 40.000 samples di 8 variabili reali.
I vari file y, come facilmente deducibile dal numero che segue y, contengono un egual numero di samples ma di 1 o più variabili.

La rete è una LSTM con uno strato denso in uscita.
Lo strato denso usa una sigmoide, quindi la rete è pensata per la regressione.
Gli ingressi e le uscite sono valori reali.

I parametri della rete e dell'addestramento sono impostabili all'interno del sorgente, nella sezione Parameters.

Una volta terminato l'addestramento viene valutato e printato l'RMSE sul testset così da avere una stima della bontà del trainig.
Lo scopo è di variare i parametri di addestramento alla ricerca del minimo dell'RMSE.

Vengono plottati due grafici: quello delle loss di training e di validazione e quello delle y e yhat.

Al termine di ogni addestramento viene anche salvato il modello, sempre con lo stesso nome.
