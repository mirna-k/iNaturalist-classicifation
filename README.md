# Klasifikacija rijetkih biljaka i životinja

Potrebno je izgraditi riješenje u Pythonu za klasifikaciju biljaka i životinja na temelju dostupnih skupova podataka (iNaturalist) te pripadnu aplikaciju. Potrebno je primjeniti koncept *one shot learning* ili *few shot learning*, gdje se model trenira da prepoznaje klasu novih primjera na temelju vrlo malo označenih primjera.

https://blog.paperspace.com/few-shot-learning/ \
https://pytorch.org/vision/stable/models.html

##### TO DO:
1. odabrati model koji je treniran za raspoznavanje značajki i uzoraka + upariti ga s sijamskom mrežom
2. napraviti dataset: Support set (otp. 10ak za svaki skup) i Query set, označiti ih ...
3. za učitavanje slika najbolje napraviti klasu s potrebnim funkcijama

#### Pripremanje slika u datasetu:
https://picsart.com/batch/
1. Crop slike na kvadrate
2. Resize na 180x180
