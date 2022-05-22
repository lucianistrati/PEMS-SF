# PEMS-SF

## Pregatirea proiectului
Inainte de toate trebuie sa descarcati datele din urmatoarele link-uri de wetransfer (cel mai tarziu pana pe 29 mai):

- PEMS: https://we.tl/t-saHInHXv02 
- PEMS-SF: WORK IN PROGRESS cu uploadarea pe wetrasnfer, dar oricum, momentan nu e nevoie de el, foarte probabil sa nu fie nevoie deloc, in esenta PEMS e o varianta mult mai simplificata si ready to go a lui PEMS-SF, daca va fi nevoie de el, se va face, dar cred ca nu
- sktime: https://we.tl/t-iJeBUtXSzk

Primul (folderul PEMS) trebuie dezarhivat in folderul "data".

Al treilea (folderul sktime) trebuie dezarhivat in root-ul proiectului.


Pentru a folosi proiectul este necesara crearea unui environment anaconda cu versiunea python 3.10:
```commandline
conda create -n my_env python==3.10
```

Pentru a activa apoi mediul:
```
conda activate my_env
```

Pentru a folosi versiunile bune ale pachetelor:
```
pip install -r requirements.txt
```
