# **ML-2023**
Repozitorijum za studentski projekat iz kursa Mašinsko učenje.

## **Učesnici na projektu**
 - Bojan Bardžić (1072/2022)

## **Problem**
Potrebno je klasifikovati skup slika karaktera u svoje odgovarajuće klase.  
Karakteri mogu biti:
- mala slova iz engleske abecede **a-z**
- velika slova iz engleske abecede **A-Z**
- cifre **0-9**

Ukupno imamo **62** klase.

## **Skup podataka**
Za ovaj projekat korišćen je skup podataka **Chars74K** koji se može preuzeti na sledećoj [stranici](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/).  
Sa stranice su preuzete sledeće datoteke:  
 - `EnglishImg.tgz`
 - `EnglishHnd.tgz`
 - `EnglishFnt.tgz`

U skupu podataka nalaze se fotografije karaktera ,slike ručno pisanih karaktera kao i slike karaktera u različitim kompjuterskim fontovima.   
Ukupno 78905 slika različitih dimenzija i fromata (rgb i grayscale).

Potrebno je raspakovati ove datoteke u folder projekta tako da na kraju
sve tri vrste slika budu u direktorijumu `English`.

## **Paketi**
Za pokretanje jupyter sveske potrebni su vam sledeći paketi:
- [**Tensorflow**](https://www.tensorflow.org/install)
- [ **Keras** ](https://keras.io/getting_started/) (dolazi zajedno sa Tensorflow-om)
- [**Numpy**](https://numpy.org/doc/stable/user/absolute_beginners.html) 
- [**Matplotlib**](https://matplotlib.org/stable/users/getting_started/)
- [**Pillow**](https://pypi.org/project/Pillow/)
- [**Scikit-learn**](https://scikit-learn.org/stable/install.html)

Takođe je potrebno imati instaliran [**Jupyter notebook**](https://jupyter.org/).

## **Pretprocesiranje i raspodela podataka**
### **Pretprocesiranje**
Svim slikama je promenjena veličina na **64x64** korišćenjem paketa **pillow** i format promenjen na **rgb**. Na kraju su slike pretvorene u **numpy** niz i preoblikovane tako da je oblik svake instance **(64, 64, 3)**.  

Takođe je opseg vrednosti piksela skaliran sa celobrojnog **[0, 255]** na opseg **[0.0, 1.0]** u pokretnom zarezu.

### **Raspodela**

Podaci su raspodeljeni u 62 klase na sledeći način:

![raspodela podataka po klasma](images/data_distribution_by_class.png) 

Može se videti da su neke klase brojnije od drugih ali da su podaci ipak ravnomerno
raspodeljeni po klasama.

Još jedna zanimljiva stvar koju možemo videti je ne samo raspodeljenost po klasama,
nego raspodeljenost po paru **(tip, klasa)**, gde je tip slika, rukopis ili font.  

![raspodela podataka po tipu i klasi](images/data_distribution_by_class_and_type.png)
  
Vidimo da u skupu podataka najviše ima slika fontova, pa onda slika koje su fotografije, dok najmanje ima slika ručno pisanih karaktera.

## **Prvi model**
### **Izbor modela**
Razmatrane su sledeće arhitekture:

Prva arhitektura:
- **Conv2D** sa **128** filtera sa ulaznim oblikom **(64, 64, 3)**
- **Conv2D** sa **64** filtera
- **MaxPool** sa velicinom pool-a **(2,2)**
- **Dropout**
- **Flatten**
- **Dense** sa **128** jedinica i **relu** aktivacijom
- **Dropout**
- **Dense** sa **62** jedinice i **softmax** aktivacijom

Druga arhitektura
- **Conv2D** sa **256** filtera sa ulaznim oblikom **(64, 64, 3)**
- **Conv2D** sa **128** filtera
- **MaxPool** sa velicinom pool-a **(2,2)**
- **Conv2D** sa **64** filtera
- **Dropout**
- **Flatten**
- **Dense** sa **128** jedinica i **relu** aktivacijom
- **Dropout**
- **Dense** sa **62** jedinice i **softmax** aktivacijom

Treća arhitektura:
- **Conv2D** sa **256** filtera sa ulaznim oblikom **(64, 64, 3)**
- **Conv2D** sa **128** filtera
- **MaxPool** sa velicinom pool-a **(2,2)**
- **Conv2D** sa **128** filtera
- **Conv2D** sa **64** filtera
- **MaxPool** sa velicinom pool-a **(2,2)**
- **Dropout**
- **Flatten**
- **Dense** sa **128** jedinica i **relu** aktivacijom
- **Dropout**
- **Dense** sa **62** jedinice i **softmax** aktivacijom

Četvrta arhitektura:
- **Conv2D** sa **64** filtera sa ulaznim oblikom **(64, 64, 3)**
- **Conv2D** sa **32** filtera
- **Conv2D** sa **16** filtera
- **MaxPool** sa velicinom pool-a **(2,2)**
- **Dropout**
- **Flatten**
- **Dense** sa **128** jedinica i **relu** aktivacijom
- **Dense** sa **64** jedinica i **relu** aktivacijom
- **Dropout**
- **Dense** sa **62** jedinice i **softmax** aktivacijom

Za konvolucione slojeve se koristi **relu** aktivacija kao i velicina kernela **(3, 3)**
i padding je podešen na **same**.

U izboru modela je takođe variran hiperparametar **dropout_rate** koji se nalazi u **Dropout** slojevima i isprobane vrednosti
su bile:
- **0.33**
- **0.5**
- **0.66**

### Podela podataka
Podaci su podeljeni na skup za **trening**, **test** i **validaciju**. Stratifikacija je vršena na odnosu para  **tip-klasa**. Od **78905** instanci iz skupa podataka, **57008** je izdvojeno za trening, **10061** za validaciju i **11836** za testiranje. 

Izbor modela je vršen na osnovu performansi na **skupu za validaciju**.

### Obučavanje modela
Treniranje je vršeno na **30** epoha sa veličinom paketića (**batch size**) od **128**. Za funkciju greške je korišćena kategorička unakrsna entropija (**categorical cross-entropy**) dok je od metrika praćena tačnost (**accuracy**). Za optimizator je korišćen **Adam** sa stopom učenja (**learning rate**) od **0.001**. Tokom obučavanja odabran je validacioni skup (ovo služi samo za validaciju tokom treniranja, razlikuje se od validacionog skupa koji se koristi za izbor modela.) koji je veličine **0.2** u odnosu na trening skup.

### Rezultati
| Model      | Tačnost | Funkcija greške |
|------------|:-------:| :--------------:|
| arch1_0.33 | 0.8404  |   0.8623        |
| arch2_0.33 | 0.8397  |   0.7629        |      
| arch3_0.33 | 0.8647  |   0.5372        |
| arch4_0.33 | 0.8368  |   0.7566        |
| arch1_0.5  | 0.8381  |   0.6089        |
| arch2_0.5  | 0.8543  |   0.5673        |
| arch3_0.5  | 0.8748  |   0.4200        |
| arch4_0.5  | 0.8413  |   0.6161        |
| arch1_0.66 | 0.8276  |   0.5838        |
| arch2_0.66 | 0.8440  |   0.5000        |
| arch3_0.66 | 0.8581  |   0.4361        |
| arch4_0.66 | 0.8272  |   0.5905        |

Na osnovu tabele zaključujemo da je najbolji model **treća arhitektura** sa verovatnoćom anuliranja u **Dropout** slojevima od **0.5** (arch3_0.5). 

### **Evaluacija modela**
#### **Tačnost i funkcija greške**
Model je evauluiran na podacima za testiranje i pokazao je:  
- Tačnost od **0.8730**
- Vrednost fukcije greške od **0.4049**

#### **Matrica konfuzije**
Dalje želimo da vidimo statistike vezane za pojedinačne klase i koliko je naš model dobar u njihovom pogađanju. Prva stvar koju ćemo pogledati je **matrica konfuzije**, koja se nalazi u direktorijumu `images\selected_architecture_30epochs_bsize_128`. Nju možemo videti na sledećoj slici:  

![confusion_matrix](images/selected_architecture_30epochs_bsize_128/selected_architecture_30epochs_bsize_128_confusion_matrix.png)  

Vidimo da naš model uglavnom dobro klasifikuje sve instance iz trening skupa. Jedna zanimljiva stvar koju možemo da vidimo su **dve paralelne dijagonale** koje se nalaze pored glavne dijagonale. Vidimo da su na njima greške mnogo češće.

 To je zato što su parovi koji čine ove dijagonale upravo **parovi veliko i malo slovo** (na primer **A i a**). To nam kaže da je model u stanju da prepozna koji se karakter nalazi na slici, ali ne može uvek da pogodi ispravno da li se radi o velikom ili o malom slovu.

 Takođe su vidljive tačke gde se cifra **0** klasifikuje kao slovo **o** i obrnuto.  

#### **Izveštaj klasifikacije**

 Sledeća stvar koju želimo da vidimo je izveštaj klasifikacije (**classification report**), on izgleda ovako:
 ``` 		
   		      precision    recall  f1-score   support

           0       0.66      0.70      0.68       183
           1       0.87      0.87      0.87       179
           2       0.97      0.97      0.97       179
           3       0.97      0.99      0.98       172
           4       0.99      0.96      0.98       171
           5       0.98      0.99      0.98       172
           6       0.97      0.96      0.97       173
           7       0.99      0.99      0.99       170
           8       0.99      0.98      0.98       167
           9       0.97      0.94      0.95       171
          10       0.94      0.96      0.95       299
          11       0.95      0.93      0.94       188
          12       0.80      0.75      0.77       212
          13       0.94      0.95      0.95       209
          14       0.89      0.96      0.92       267
          15       0.93      0.95      0.94       181
          16       0.93      0.91      0.92       194
          17       0.92      0.94      0.93       207
          18       0.78      0.77      0.78       241
          19       0.92      0.92      0.92       176
          20       0.95      0.92      0.94       181
          21       0.91      0.96      0.94       211
          22       0.90      0.95      0.93       199
          23       0.91      0.94      0.93       244
          24       0.66      0.43      0.52       248
          25       0.92      0.88      0.90       200
          26       0.92      0.93      0.93       167
          27       0.92      0.94      0.93       253
          28       0.75      0.79      0.77       242
          29       0.88      0.95      0.91       238
          30       0.86      0.86      0.86       185
          31       0.71      0.85      0.77       179
          32       0.81      0.74      0.78       178
          33       0.75      0.73      0.74       174
          34       0.95      0.93      0.94       176
          35       0.77      0.68      0.72       170
          36       0.92      0.92      0.92       208
          37       0.95      0.94      0.95       169
          38       0.72      0.79      0.75       177
          39       0.96      0.91      0.94       177
          40       0.94      0.88      0.91       220
          41       0.96      0.93      0.95       169
          42       0.92      0.90      0.91       171
          43       0.95      0.91      0.93       173
          44       0.93      0.92      0.93       198
          45       0.99      0.90      0.94       166
          46       0.94      0.93      0.94       169
          47       0.74      0.79      0.76       182
          48       0.93      0.95      0.94       173
          49       0.88      0.92      0.90       198
          50       0.55      0.76      0.64       200
          51       0.90      0.89      0.89       172
          52       0.91      0.93      0.92       168
          53       0.93      0.88      0.91       198
          54       0.72      0.64      0.68       188
          55       0.91      0.89      0.90       192
          56       0.87      0.85      0.86       172
          57       0.80      0.62      0.70       167
          58       0.75      0.82      0.78       168
          59       0.70      0.74      0.72       168
          60       0.88      0.91      0.90       171
          61       0.72      0.77      0.75       166

    accuracy                           0.87     11836
   macro avg       0.88      0.87      0.87     11836
weighted avg       0.87      0.87      0.87     11836
```

Možemo primetiti da su vrednosti **f1 mere** visoke za većinu klasa dok je su najniže vrednosti oko **60%**, što nije loš rezultat. U proseku naša mreža ima preciznost od **0.87** i odziv od **0.87**.

#### **Zanemarivanje grešaka velikih i malih slova**

Još jedna napredna statistika koju možemo izračunati je preciznost kada **ne bismo računali greške malih i velikih slova**, kao i greške mešanja cifre **0** i karaktera '**o**'/'**O**'. Onda bi naš model imao:
- Tačnost od **0.9507**.

Ako želimo da budemo strožiji možemo da priznajemo samo greške onih slova gde su velika i mala slova skoro identična. To su **C**, **I**, **J**, **K**, **O**, **P**, **S**, **U**, **V**, **W**, **X**, **Y** i **Z**.

Opet ćemo priznatvati greške sa nulom i slovom **O**. U ovom slučaju dobijamo:
- Tačnost od **0.9146**

#### Zaključak evaluacije kvaliteta modela

Zaključujemo da iako model greši na nekim mestima on generalno pokazuje dobre rezultate prepoznavanja karaktera.

## Drugi model
### Arhitektura modela
Drugi model koristi arhitekturu **Alexnet** koja se može pronaći na sledećoj stranici:
https://www.analyticsvidhya.com/blog/2021/03/introduction-to-the-architecture-of-alexnet/  

Jedina razlika na originalni model sto je u prvom sloju parametar **stride** smanjen
sa (4, 4) na (1, 1)

Model se sastoji od sledećih slojeva:
- **Conv2D** sa **96** filtera, veličinom kernela **(11, 11)** i ulaznim oblikom (64, 64, 3)
- **MaxPool** sa veličinom pool-a **(3, 3)** i stride-om veličine **2**
- **Conv2D** sa **256** filtera, veličinom kernela **(5, 5)**, stride-om **1**
- **MaxPool** sa veličinom pool-a **(3, 3)** i stride-om veličine **2**
- **Conv2D** sa **384** filtera, veličinom kernela **(3, 3)**, stride-om **1**
- **Conv2D** sa **384** filtera, veličinom kernela **(3, 3)**, stride-om **1**
- **Conv2D** sa **256** filtera, veličinom kernela **(3, 3)**, stride-om **1**
- **MaxPool** sa veličinom pool-a **(3, 3)** i stride-om veličine **2**
- **Dropout** sa verovtnoćom anuliranja **0.5**
- **Dense** sa **4096** jedinica
- **Dropout** sa verovtnoćom anuliranja **0.5**
- **Dense** sa **4096** jedinica
- **Dense** sa **62** jedinice i **softmax** aktivacionom funkcijom

U svim **Conv2D** i **Dense** slojeva korišćena je **relu** aktivaciona funkcija.  
U svim **Conv2D** slojevima je parametar padding podešen na **'same'**.

### **Obučavanje modela**
Obučavanje je vršeno na trening podacima na 50 epoha sa veličinom paketića 128. Vrednosti funkcije greške i tačnosti na test i validacionim skupovima kroz epohe obučavanja se mogu videti na sledećoj slici:

![training_stats2](images/architecture2_50epochs_bsize_128/architecture2_50epochs_bsize_128_2.png)

Vidimo da nema velikog preprilagođavanja, ali dok performanse na trening skupu blago rastu kroz epohe, performanse na validacionom skupu počinju da stagniraju posle **10.** epohe.

### **Evaluacija modela**
#### **Tačnost i funkcija greške**
Na test podacima drugi model je pokazao:
- Tačnost od **0.8778**
- Vrednost fukcije greške od **0.3853**


#### **Matrica konfuzije**
Sada možemo pogledati i napredne statistike za naš drugi model. Matrica klasifikacije izgleda ovako:

![confusion_matrix2](images/architecture2_50epochs_bsize_128/architecture2_50epochs_bsize_128_confusion_matrix_2.png)

Možemo videti da i dalje imamo slične probleme kao sa prvim modelom, a to je razlikovanje između velikih i malih slova.

#### **Izveštaj klasifikacije**
Dalje možemo videti izveštaj klasifikacije:

```
   	  		precision    recall  f1-score   support

           0       0.64      0.75      0.69       183
           1       0.86      0.92      0.89       179
           2       0.98      0.97      0.97       179
           3       0.96      0.99      0.97       172
           4       0.99      0.96      0.98       171
           5       0.99      0.98      0.99       172
           6       0.97      0.99      0.98       173
           7       0.98      0.98      0.98       170
           8       0.99      0.98      0.98       167
           9       0.95      0.94      0.94       171
          10       0.94      0.94      0.94       299
          11       0.95      0.96      0.96       188
          12       0.75      0.86      0.80       212
          13       0.90      0.95      0.92       209
          14       0.92      0.97      0.95       267
          15       0.91      0.94      0.93       181
          16       0.93      0.93      0.93       194
          17       0.95      0.92      0.93       207
          18       0.84      0.77      0.81       241
          19       0.94      0.95      0.94       176
          20       0.90      0.97      0.93       181
          21       0.94      0.93      0.94       211
          22       0.91      0.96      0.94       199
          23       0.96      0.94      0.95       244
          24       0.67      0.52      0.59       248
          25       0.92      0.88      0.90       200
          26       0.92      0.94      0.93       167
          27       0.95      0.96      0.96       253
          28       0.76      0.65      0.70       242
          29       0.95      0.94      0.95       238
          30       0.87      0.93      0.90       185
          31       0.67      0.88      0.76       179
          32       0.84      0.68      0.75       178
          33       0.67      0.83      0.74       174
          34       0.94      0.98      0.96       176
          35       0.75      0.74      0.74       170
          36       0.92      0.91      0.92       208
          37       0.98      0.95      0.96       169
          38       0.81      0.67      0.73       177
          39       0.93      0.93      0.93       177
          40       0.96      0.90      0.93       220
          41       0.95      0.94      0.94       169
          42       0.92      0.89      0.90       171
          43       0.96      0.91      0.93       173
          44       0.95      0.90      0.92       198
          45       0.97      0.93      0.95       166
          46       0.96      0.92      0.94       169
          47       0.75      0.75      0.75       182
          48       0.89      0.97      0.93       173
          49       0.91      0.92      0.91       198
          50       0.59      0.67      0.62       200
          51       0.92      0.87      0.89       172
          52       0.92      0.93      0.93       168
          53       0.94      0.91      0.93       198
          54       0.62      0.78      0.69       188
          55       0.92      0.94      0.93       192
          56       0.94      0.85      0.89       172
          57       0.79      0.54      0.64       167
          58       0.74      0.85      0.79       168
          59       0.75      0.57      0.64       168
          60       0.95      0.92      0.94       171
          61       0.74      0.77      0.76       166

    accuracy                           0.88     11836
   macro avg       0.88      0.88      0.88     11836
weighted avg       0.88      0.88      0.88     11836
```

Vidimo da imamo uglavnom visoke vrednosti za **f1-score** i **preciznost**. Prosečna preciznost je **0.88** dok je prosečan odziv **0.88**.

#### **Zanemarivanje grešaka velikih i malih slova**
Možemo videti i tačnost modela gde zanemarujemo greške velikih i malih slova kao i nule i slova **O**. Ako zanemarimo sve takve greške dobijamo:
- tačnost od **0.9570**

I ako zanemarimo samo ona slična slova dobijamo:
- tačnost od **0.9154**

### **Poređenje sa prvim modelom**

Na podacima za **validaciju** ovaj model ima sledeće rezultate:
- Tačnost od: **0.8696**
- Vrednost funkcije greške od: **0.43195**

Što je lošije od prvog modela iako je drugi model kompleksniji.

## **Zaključak**

Oba modela daju zadovoljavajuće rezultate, razlika između njihovih performansi je jako mala.

## **Linkovi do modela**

Ukoliko želite da preuzmete modele koje sam trenirao, njih možete pronaći na mom google drive nalogu.
- Link do prvog modela: https://drive.google.com/drive/folders/1ZUdeuCb88ZSce0piRfgJQutz8AwbeOdQ?usp=sharing
- Link do drugog modela: https://drive.google.com/drive/folders/1rykeQ1an3RZCLKm9idaGE61VPE0nAFno?usp=sharing

Kada preuzmete modele, njih možete učitati pozivom metoda:  
 **keras.models.load_model(*putanja_do_direktorijuma_u_kom_je_model*)**
