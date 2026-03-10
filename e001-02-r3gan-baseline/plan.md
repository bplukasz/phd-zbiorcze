Po przejrzeniu starej roadmapy i Twojego kodu mam taki wniosek:

**kierunek wavelet/frequency-aware ma sens naukowy, ale stara roadmapa była zbyt mocno „paper-first”, za mało „doktorat-first”.**
Najbardziej obroniona wersja tego pomysłu to nie „zbuduj nowego GAN-a z waveletami”, tylko:

**dodaj kontrolowany bias częstotliwościowy do Twojego obecnego baseline’u R3GAN i pokaż, że efekt nie wynika tylko z większej pojemności modelu, innej procedury treningowej albo innego budżetu obliczeniowego.**

To jest szczególnie ważne dlatego, że Twój aktualny baseline nie jest „oficjalnym kodem R3GAN”, tylko własną, spójną implementacją z własnym szkieletem generatora/dyskryminatora, relatywnym losssem, zero-centered gradient penalty dla real i fake, EMA oraz nowoczesnym training loopem. Roadmapa powinna więc **rozszerzać ten kod modularnie**, a nie podmieniać go na inny GAN tylko dlatego, że tak było wygodniej w starej wersji. 

Dodatkowo sam kierunek jest sensowny merytorycznie: R3GAN był przedstawiany jako nowoczesny, uproszczony baseline oparty na regularized relativistic loss, a prace o biasie częstotliwościowym pokazują, że problemy z wysokimi częstotliwościami i sygnałem z dyskryminatora są realne oraz że poprawa części dyskryminacyjnej jest sensownym kierunkiem badawczym. Jednocześnie samo „użyłem waveletów” nie jest już nowością — podobne motywy pojawiały się już w pracach o częstotliwościowych/waveletowych stratach i dyskryminatorach, więc wkład trzeba zbudować na **sposobie integracji, kontroli eksperymentalnej i jakości analizy**, a nie na samym DWT. ([openreview.net][1])

## 1. Trzy sensowne opcje

### Opcja A — rekomendowana

**Wavelet-aware discriminator + lekka regularizacja statystyk HF po stronie generatora + mocne eksperymenty kontrolne**

To jest najlepsza opcja, gdy chcesz:

* zachować uczciwość względem baseline’u R3GAN,
* mieć realną szansę na pozytywny wynik,
* dać temu formę sensownego rozdziału doktorskiego albo publikacji.

**Minimalna wersja do szybkiego testu**

* tylko gałąź waveletowa w dyskryminatorze,
* tylko Haar, poziom 1,
* tylko pasma HF: LH/HL/HH,
* fuzja po pierwszym downsamplingu w D,
* bez zmiany generatora,
* bez wavelet output.

**Pełna wersja**

* Wavelet branch w D,
* generator-only wavelet statistic matching,
* kontrola FFT,
* kontrola „matched-capacity” bez waveletów,
* 2 datasety i finalnie 3 seedy.

To jest najbardziej naukowo sensowne.

---

### Opcja B — bardziej konserwatywna

**Gałąź metodologiczna: diagnostyka częstotliwościowa + protokół stabilności i jakości, bez zmiany architektury**

To ma sens, gdy:

* chcesz najpierw zbudować porządne środowisko oceny,
* nie jesteś pewien, czy sama modyfikacja modelu da wyraźny zysk,
* zależy Ci na materiale do rozdziału o metrykach/diagnostyce GAN.

Tutaj wkładem nie jest nowa architektura, tylko:

* protokół porównawczy względem R3GAN,
* sensowne pomocnicze miary częstotliwościowe,
* analiza stabilności,
* rozdzielenie „realnej poprawy rozkładu” od „estetyki HF”.

To jest słabsze jako „nowa metoda”, ale mocne metodologicznie.

---

### Opcja C — wysoki risk / wysoki koszt

**Wavelet-output generator albo learnable wavelets**

To ma sens tylko wtedy, gdy:

* opcja A już działa,
* chcesz zrobić eksperyment rozszerzający,
* jesteś gotowy na większe ryzyko interpretacyjne.

Problem tej opcji jest prosty:

* mieszasz zmianę reprezentacji wyjścia generatora,
* zmianę architektury,
* zmianę przepływu gradientów,
* często też zmianę pojemności modelu.

To dużo trudniej obronić jako uczciwe porównanie z R3GAN.
Do doktoratu nadaje się raczej jako **drugi etap**, nie jako pierwszy.

---

## 2. Co bym zmienił względem starej roadmapy

Najważniejsze zmiany:

### 1) Nie zmieniałbym baseline’u na „ResNet GAN + hinge + SN”

To było sensowne pod szybki paper, ale **nie ma sensu metodologicznego** w Twoim obecnym kontekście.
Masz już własny R3GAN-like baseline i to on ma być stałym punktem odniesienia. 

### 2) Nie dawałbym regularizacji waveletowej do `loss_D`

W starej roadmapie było „do G i/lub D”.
Ja bym to uciął:

* **wavelet branch** → w dyskryminatorze,
* **wavelet stat regularizer** → tylko do generatora.

Dlaczego:

* dyskryminator już ma bardzo silny sygnał treningowy,
* w Twoim kodzie D-loss i tak jest relatywnie ciężki przez gradient penalty na real i fake,
* dokładanie tam jeszcze statystycznej regularizacji zwiększa ryzyko niestabilności i kosztu.

### 3) Nie zaczynałbym od db2 ani L=2

Na start:

* tylko `haar`,
* tylko `level=1`,
* tylko `HF-only`.

db2 i L=2 to ablacjki później.
Na początku one bardziej zwiększają ryzyko implementacyjnego bałaganu niż wartość naukową.

### 4) Dodałbym koniecznie kontrolę „matched-capacity”

Bez tego recenzent może powiedzieć:

> „To nie wavelety pomogły, tylko dodatkowa gałąź i większa pojemność dyskryminatora.”

Czyli potrzebujesz wariantu kontrolnego:

* dokładnie taka sama dodatkowa gałąź,
* taki sam koszt i liczba parametrów,
* ale zamiast DWT bierzesz np. `avgpool(RGB)` albo prosty strided-conv branch.

To jest **bardzo ważne**. W starej roadmapie tego brakowało.

### 5) Porównania robiłbym po **kimg / images seen**, nie po samych iteracjach

Przy różnych batch size porównanie po iteracjach jest metodologicznie słabe.
Raportuj:

* FID/KID/P/R vs **kimg**,
* finalny wynik po tym samym budżecie obrazów,
* czas i VRAM osobno.

---

## 3. Rekomendowana roadmapa — wersja docelowa

To jest wersja, którą bym realnie wdrażał.

## Etap 0. Zamrożenie baseline’u R3GAN

Najpierw ustalasz **nienaruszalny baseline eksperymentalny**:

* ten sam loss,
* te same optimizer settings,
* to samo EMA,
* ten sam preprocessing,
* ten sam augment,
* ten sam scheduler lub jego brak,
* ten sam budżet treningu wyrażony w kimg,
* te same real stats do FID/KID,
* te same zestawy latentów do sample grids.

W Twoim kodzie domyślne ustawienia treningowe są już sensownym punktem startowym: Adam `lr_g=lr_d=2e-4`, `betas=(0.0, 0.99)`, `gamma=10.0`, `ema_beta=0.999`, channels-last, AMP głównie dla generatora. Tego bym nie ruszał w tej gałęzi badań, chyba że cała praca dotyczyłaby właśnie procedury treningowej. 

### Co dopisać od razu

* `exp_name`
* `seed`
* `dataset_name`
* `resolution`
* `kimg_budget`
* `eval_every_kimg`
* `fixed_eval_latents.pt`
* `metrics.csv`
* `train_log.csv`
* zapis pełnego configu do YAML/JSON
* hash commita / wersję kodu

To nie jest drobiazg techniczny. To jest warunek reprodukowalności.

---

## Etap 1. Zbudowanie otoczki ewaluacyjnej

Twój plik zawiera model i pętlę treningową, ale nie zawiera jeszcze pełnej otoczki badawczej: runnera eksperymentów, kontrolowanej ewaluacji, metryk i ablation harness. To trzeba dobudować najpierw. 

### Metryki główne

Zgodnie z Twoim profilem:

* **FID**
* **KID**
* **Precision / Recall**
* **LPIPS**, ale traktowany jako **miara różnorodności/perceptual spread**, nie jako główna miara jakości realizmu

To ostatnie jest ważne: przy bezwarunkowym GAN LPIPS sam w sobie nie mówi „ten model jest lepszy”, tylko bardziej „jak bardzo zróżnicowane perceptualnie są próbki”.

### Metryki pomocnicze, które tutaj mają sens

Tylko dwie:

* **RPSE** — różnica radial power spectrum,
* **WBED** — dystans statystyk energii pasm waveletowych.

Nie traktuj ich jako „metryk jakości końcowej”, tylko jako **diagnostykę mechanizmu**.

### Co logować co krok / co okno

* `d_loss`, `g_loss`
* `r1`, `r2` lub ich odpowiedniki z Twojego kodu
* `real_score_mean`, `fake_score_mean`
* `grad_norm_G`, `grad_norm_D`
* `time_per_step`
* `images_per_sec`
* `max_memory_allocated`
* `ema_decay`
* `lambda_wave`
* surowa wartość `wave_reg_raw`
* surowa wartość `fft_reg_raw`

### Co liczyć co ewaluację

* FID
* KID
* P/R
* LPIPS-diversity
* RPSE
* WBED
* sample grid z tych samych latentów
* optionally nearest neighbors dla kilku próbek

---

## Etap 2. Implementacja DWT/iDWT — ale rozsądnie

### Mój wybór na start

* tylko **Haar**
* tylko **level = 1**
* brak „sprytnych” learnable filters
* brak db2 na początku

### Dlaczego

Haar daje:

* prostotę,
* łatwą interpretację,
* małe ryzyko błędu implementacyjnego,
* dobrą kontrolę kierunkowych HF.

### Minimalna specyfikacja modułu

Osobny plik, np. `wavelets.py`

```python
class FixedHaarDWT2d(nn.Module):
    def forward(self, x) -> dict[str, torch.Tensor]:
        # x: [B, C, H, W]
        # returns:
        # {
        #   "LL": [B, C, H/2, W/2],
        #   "LH": [B, C, H/2, W/2],
        #   "HL": [B, C, H/2, W/2],
        #   "HH": [B, C, H/2, W/2],
        # }
```

### Implementacyjnie

Najprościej:

* 4 stałe filtry 2x2 jako `buffers`,
* `groups=C`,
* `conv2d(stride=2)`.

Dla Haar przy parzystych rozdzielczościach **nie potrzebujesz paddingu** do podstawowego wariantu. To jest prostsze i czystsze niż od razu pchać `reflect`, które bywa źródłem przesunięć fazowych.
Padding dorzucałbym dopiero przy dłuższych filtrach typu db2.

### Testy obowiązkowe

1. `IDWT(DWT(x)) ≈ x`
2. `max_abs_err`
3. `MAE`
4. `PSNR`
5. test na tensorze losowym
6. test na realnym batchu obrazów

Jeśli ten etap nie jest perfekcyjny, dalej nie idziesz.

---

## Etap 3. Wavelet branch w dyskryminatorze — dokładna wersja

To jest główny element metody.

### Gdzie wpiąć branch

**Po pierwszym downsamplingu w D**, a nie na samym wejściu i nie jako osobny pełny drugi dyskryminator.

To jest najczystsze rozwiązanie dla Twojej architektury.

W praktyce:

1. główny tor:

```python
x0 = from_rgb(x)          # H x W
x1 = stages[0](x0)        # H/2 x W/2
```

2. tor waveletowy:

```python
bands = dwt(x)
wave = cat([LH, HL, HH], dim=1)   # HF-only
wave_feat = branch(wave)          # -> shape zgodny z x1
```

3. fuzja:

```python
x1 = x1 + gate * wave_feat
```

4. dalej normalnie:

```python
for stage in stages[1:]:
    x1 = stage(x1)
```

### Dlaczego tak

* pasuje do obecnej geometrii D,
* nie rozwala architektury,
* nie zwiększa drastycznie kosztu,
* pozwala w miarę uczciwie izolować efekt.

### Jak ma wyglądać branch

Nie rób pełnej wieży. Zrób mały branch:

```python
wave_in_ch = 3 * in_channels  # LH, HL, HH
mid_ch = stage_channels[1] // 2
out_ch = stage_channels[1]

wave_branch:
    Conv2dNoBias(wave_in_ch, mid_ch, 3)
    BiasAct2d(mid_ch)
    Conv2dNoBias(mid_ch, out_ch, 3)
```

### Bardzo ważny detal

* ostatnią konwolucję brancha wyzeruj na starcie,
* `gate` ustaw jako uczony skalar z inicjalizacją `0.0`.

Czyli model startuje **dokładnie jak baseline**, a branch „wchodzi do gry” dopiero jeśli trening uzna go za użyteczny.

To jest świetne metodologicznie i praktycznie.

### Czego nie robić

* nie robić concat + duży 1x1 + duża wieża, bo trudniej kontrolować pojemność,
* nie robić od razu multi-level fusion,
* nie dawać LL w pierwszej wersji.

---

## Etap 4. Wariant kontrolny „matched-capacity”

Musisz mieć taki wariant.

### Konstrukcja

Dokładnie ten sam branch i ta sama fuzja, ale wejście nie pochodzi z DWT, tylko np. z:

* `avg_pool2d(x, 2)` + projekcja kanałów,
  albo
* prosty strided-conv branch z RGB.

Ten eksperyment odpowiada na pytanie:

> czy zysk daje waveletowa reprezentacja, czy po prostu dodatkowa ścieżka?

Bez tego wkład jest dużo słabszy.

---

## Etap 5. Wavelet-stat regularization — tylko do generatora

To jest drugi element metody.

### Główna zasada

Nie dopasowuj pikseli.
Nie rób paired loss.
Nie wciskaj tego do D-loss.

Rób:

* batch-level,
* distribution matching,
* tylko po stronie generatora.

### Jakie statystyki

Dla każdego obrazu:

```python
E_LH = mean(LH**2)
E_HL = mean(HL**2)
E_HH = mean(HH**2)
```

Lepiej jeszcze:

```python
logE = log(eps + E)
```

Bo:

* skala jest stabilniejsza,
* LL nie dominuje,
* wartości są bardziej porównywalne.

### Co porównywać

Po batchu:

* `mu_fake`, `std_fake`
* `mu_real_ref`, `std_real_ref`

i loss:

```python
L_wave = L1(mu_fake, mu_ref) + 0.5 * L1(std_fake, std_ref)
```

### Skąd brać `mu_ref`, `std_ref`

Nie z bieżącego batcha „w locie” przy każdym forwardzie jako twardego celu, tylko lepiej:

* utrzymywać **EMA statystyk realnych** jako bufor.

Przykład:

```python
mu_ref = 0.99 * mu_ref + 0.01 * mu_real_batch
std_ref = 0.99 * std_ref + 0.01 * std_real_batch
```

To bardzo pomaga, szczególnie przy mniejszych batchach.

### Gdzie to wpiąć

Nie pchaj tego do `R3GANLoss.generator_loss()` w obecnej postaci bez refaktoru, bo wtedy łatwo zrobisz:

* podwójny forward generatora,
* podwójną augmentację,
* niepotrzebny koszt.

Lepiej zrobić tak:

1. w kroku G:

```python
fake = G(z, cond)
```

2. policzyć adv loss ręcznie na tym `fake`
3. policzyć `wave_reg(fake)`
4. zsumować:

```python
g_total = g_adv + lambda_wave * g_wave
```

To jest czystsze niż dokładanie tego „na siłę” do obecnej funkcji `generator_loss`.
W Twoim kodzie właśnie ten etap wymaga małego refaktoru training stepu. 

### Startowe wartości

Dla log-energy formulation:

* start: `lambda_wave = 0.02`
* sweep: `{0.005, 0.02, 0.05}`

Ale ważniejsza od samej liczby jest zasada:

* na początku `lambda_wave * L_wave_raw` ma być około **5–10%** `g_adv`.

To jest lepsze niż ślepe zgadywanie.

---

## Etap 6. FFT baseline kontrolny

To ma sens i bym to zostawił.

Ale w wersji czystej:

* dokładnie ten sam styl regularizacji,
* też tylko do generatora,
* radial bins,
* też log-energy,
* też mean/std matching.

Wtedy naprawdę porównujesz:

* czy pomaga sam frequency prior,
* czy konkretnie waveletowy.

### Ustawienia

* 16 albo 24 radial bins
* pomiń DC
* możesz rozważyć liczenie tylko wyższych binów jako HF-control

---

## Etap 7. Plan eksperymentalny

### Faza A — szybki test wykonalności

Cel: sprawdzić, czy kierunek żyje.

1 dataset, 1 rozdzielczość, 1 seed

Porównujesz:

* `R0`: baseline
* `R1`: baseline + matched-capacity branch
* `R2`: baseline + WaveD(HF)
* `R3`: baseline + WaveD(HF) + WaveReg
* `R4`: baseline + WaveD(HF) + FFTReg

Na tym etapie:

* jeszcze bez db2,
* bez level 2,
* bez wavelet output.

### Faza B — eksperyment rozwojowy

1 dataset, 1 rozdzielczość, 2 seedy

Celem jest wybrać jeden finalny wariant do potwierdzenia.

### Faza C — eksperyment potwierdzający

Finalnie:

* baseline vs najlepszy wariant
* 3 seedy
* 2 datasety **albo** 1 dataset + 2 rozdzielczości

Mój wybór:

* najpierw **2 datasety przy 128**
* dopiero potem wybrany test na **256**

To daje lepszą obronę niż szybkie skakanie po wielu rozdzielczościach.

---

## 4. Ustawienia krytyczne dla uczciwego porównania z R3GAN

To są rzeczy, których nie wolno rozjechać między wariantami:

* ten sam loss adversarialny,
* ten sam `gamma`,
* ten sam EMA,
* ten sam optimizer i bety,
* ten sam preprocessing,
* ta sama augmentacja,
* ten sam batch w danym porównaniu,
* ten sam budżet **kimg**,
* ten sam harmonogram ewaluacji,
* te same real features/statystyki do FID/KID,
* te same seedy dla porównań finalnych,
* ten sam kod generowania próbek eval.

Dodatkowo raportuj:

* liczbę parametrów,
* czas kroku,
* peak VRAM.

Bo inaczej poprawa może być „kupiona” kosztem dużo większego modelu.

---

## 5. Jak oceniać wynik

## Metryki główne

* **FID**
* **KID**
* **Precision / Recall**
* **LPIPS-diversity**

## Metryki pomocnicze

* **RPSE**
* **WBED**

## Co ma oznaczać „realna poprawa”

Za realną poprawę uznawałbym sytuację, w której:

1. zysk w FID/KID jest powtarzalny między seedami,
2. Precision/Recall nie pokazuje wyraźnego pogorszenia pokrycia rozkładu,
3. LPIPS-diversity nie spada dramatycznie,
4. pomocnicze metryki HF poprawiają się zgodnie z tezą,
5. matched-capacity control nie osiąga tego samego.

To jest ważne:
**jeśli poprawia się tylko WBED/RPSE, a FID/KID/P/R stoją albo się pogarszają, to nie masz jeszcze poprawy GAN-a — masz poprawę diagnostyki częstotliwościowej.**

## Sanity-checki

Obowiązkowo:

* `IDWT(DWT(x))` rekonstrukcja,
* czy `gate=0` daje baseline,
* czy branch z wyzerowaną końcówką naprawdę nie zmienia startu,
* czy `wave_reg` nie rośnie tylko przez dodanie szumu HF,
* wizualizacja pasm LH/HL/HH dla real/fake,
* nearest-neighbor check dla kilku próbek,
* ten sam budżet kimg.

## Testy statystyczne

Najuczciwiej:

* raportuj **mean ± std po seedach**,
* KID z własnym estymatorem i CI,
* dla głównych porównań możesz dać test parowany po seedach, ale przy 3 seedach nie robiłbym z tego wielkiej historii.

Tu bardziej liczy się:

* zgodność kierunku efektu,
* wielkość efektu,
* stabilność między seedami.

## Dodatkowa metryka stabilności

Dodałbym:

* **AUC FID-vs-kimg**,
* oraz **run success rate**.

Czyli nie tylko najlepszy punkt, ale też:

* jak szybko model dochodzi do sensownego poziomu,
* jak często trening kończy się używalnym checkpointem.

To jest dużo lepsze dla rozdziału o stabilności.

---

## 6. Krótki szkic do rozprawy

### Sens badawczy

Celem eksperymentu jest zbadanie, czy wprowadzenie jawnej informacji częstotliwościowej do procesu dyskryminacji i regularizacji generatora poprawia jakość oraz stabilność trenowania GAN względem baseline’u R3GAN, bez zmiany podstawowego celu adversarialnego.

### Hipoteza

Jawnie częstotliwościowy sygnał pomocniczy, wprowadzony w kontrolowany sposób po stronie dyskryminatora oraz jako lekka regularizacja statystyk HF po stronie generatora, poprawi zgodność generowanych obrazów z rozkładem danych rzeczywistych w zakresie wysokich częstotliwości, a jednocześnie nie pogorszy jakości mierzonej standardowymi metrykami generatywnymi.

### Opis eksperymentu

Porównano baseline R3GAN z wariantami rozszerzonymi o:
(1) dodatkową gałąź waveletową w dyskryminatorze,
(2) regularizację statystyk energii pasm HF po stronie generatora,
(3) odpowiednie warianty kontrolne o porównywalnej pojemności i koszcie obliczeniowym.

### Interpretacja wyniku

Jeżeli poprawa utrzymuje się w FID/KID oraz w metrykach pomocniczych, przy zachowaniu Precision/Recall i bez istotnego wzrostu kosztu obliczeniowego, można ją interpretować jako dowód, że częstotliwościowy bias stanowi użyteczne rozszerzenie baseline’u R3GAN. Jeżeli zysk dotyczy wyłącznie metryk pomocniczych, wynik należy interpretować ostrożnie jako poprawę zgodności częstotliwościowej, a nie jednoznaczną poprawę jakości modelu.

### Ograniczenia

Metoda może być zależna od typu danych, rozdzielczości i sposobu downsamplingu w dyskryminatorze. Dodatkowo zgodność częstotliwościowa nie musi w pełni przekładać się na poprawę jakości semantycznej.

---

## 7. Ryzyka, pułapki i jak je ograniczyć

### Ryzyko 1: „pomaga, bo ma więcej parametrów”

Ograniczenie:

* matched-capacity control,
* raport liczby parametrów i czasu.

### Ryzyko 2: poprawa RPSE/WBED przez sztuczny szum HF

Ograniczenie:

* Precision/Recall,
* LPIPS-diversity,
* inspekcja wizualna,
* nearest-neighbor check,
* kontrola na FFT regularizer.

### Ryzyko 3: branch destabilizuje D

Ograniczenie:

* HF-only,
* level 1,
* gate init = 0,
* zero-init ostatniej conv.

### Ryzyko 4: za dużo eksperymentów naraz

Ograniczenie:

* najpierw tylko Haar + HF-only + 1 miejsce fuzji,
* później dopiero ablacjki:

  * all-bands,
  * db2,
  * level 2.

### Ryzyko 5: mylenie „stabilniejszego wykresu loss” z realną stabilnością

Ograniczenie:

* AUC FID-vs-kimg,
* success rate,
* variance po seedach,
* final metrics.

### Ryzyko 6: mieszanie gałęzi badawczych

Nie mieszałbym tutaj jeszcze:

* CLIP/DINO/I-JEPA,
* semantycznych metryk,
* transformerów.

To jest osobna oś doktoratu.
Tutaj lepiej mieć **czysty eksperyment frequency-aware**.

---

## 8. Ocena innowacyjności

### Czy to może być wkład naukowy?

**Tak, ale nie w wersji „wrzuciłem DWT do GAN”.**

Taki wkład staje się obroniony dopiero wtedy, gdy pokażesz:

1. że integracja z R3GAN jest kontrolowana i uczciwa,
2. że efekt przechodzi przez matched-capacity control,
3. że poprawa jest powtarzalna między seedami,
4. że nie kończy się na pomocniczych metrykach HF,
5. że koszt obliczeniowy jest jawnie raportowany,
6. że wynik utrzymuje się na więcej niż jednym ustawieniu danych.

### Kiedy to jest tylko dobra inżynieria

Gdy:

* poprawa jest marginalna,
* znika po matched-capacity control,
* działa tylko na jednej twarzo-centrycznej bazie,
* poprawia tylko własne metryki pomocnicze.

### Co musiałoby zostać pokazane, żeby to obronić jako wkład badawczy

Najlepsza narracja byłaby taka:

* R3GAN jest nowoczesnym, dobrze ugruntowanym baseline’em. ([openreview.net][1])
* Wiadomo, że generatywne modele mają problemy z biasem częstotliwościowym, a ograniczenia po stronie dyskryminatora są realnym tropem. 
* Sama idea frequency-aware / wavelet-aware nie jest nowa, więc wkład nie polega na samym DWT, tylko na:

  * **minimalistycznej integracji z R3GAN,**
  * **mocnym protokole kontrolnym,**
  * **diagnostyce stabilności i HF,**
  * **pokazaniu, kiedy to pomaga, a kiedy nie.**

To już jest sensowny materiał doktorski.

---

## 9. Mój werdykt praktyczny

**Tak — ta gałąź ma sens i ma szansę powodzenia.**
Ale w wersji, którą bym rekomendował, wygląda to tak:

### Wersja minimalna

* baseline R3GAN bez zmian,
* WaveD(HF-only, Haar, level 1, fuse after first D stage),
* matched-capacity control,
* 1 dataset, 1 rozdzielczość, 1 seed,
* FID/KID/P/R/LPIPS + RPSE/WBED.

### Wersja pełna

* * generator-only WaveReg,
* * FFTReg control,
* 2 datasety,
* 3 seedy finalne,
* raport po kimg,
* AUC FID-vs-kimg,
* koszt i VRAM,
* ablacjki tylko dla wybranego wariantu.

### Czego bym nie robił teraz

* wavelet-output generator jako główny wątek,
* learnable wavelets,
* db2/L=2 od pierwszego dnia,
* mieszania tego z CLIP/DINO,
* przechodzenia na inny baseline niż Twój obecny R3GAN. 

W następnej wiadomości mogę Ci to przepisać w formie **bardzo konkretnej roadmapy implementacyjnej dzień-po-dniu / etap-po-etapie**, już z nazwami klas, funkcji, configów i kolejnością commitów, tak żeby drugi model mógł to praktycznie wdrażać bez zgadywania.

[1]: https://openreview.net/forum?id=OrtN9hPP7V "The GAN is dead; long live the GAN! A Modern GAN Baseline | OpenReview"
