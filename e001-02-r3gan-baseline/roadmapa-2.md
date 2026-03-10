Roadmapa badań: frequency-aware rozszerzenie
baseline'u R3GAN
Zestawienie i uporządkowanie dwóch odpowiedzi: ocena sensu naukowego oraz nowa
roadmapa implementacyjno-badawcza
Wersja robocza PDF przygotowana na podstawie odpowiedzi z 8 marca 2026 r.
Cel dokumentu: uporządkować ocenę kierunku badawczego oraz przełożyć ją na konkretną,
wykonalną i metodologicznie uczciwą roadmapę względem Twojego własnego baseline'u R3GAN.
Część I. Ocena poprzedniej roadmapy i rekomendowany kierunek
Po przejrzeniu starej roadmapy i kodu wyjściowego wniosek jest następujący: kierunek
wavelet/frequency-aware ma sens naukowy, ale stara roadmapa była zbyt mocno paper-first, a za
mało doktorat-first.
Najbardziej obroniona wersja tego pomysłu to nie budowa całkiem nowego GAN-a z waveletami,
tylko dodanie kontrolowanego biasu częstotliwościowego do obecnego baseline'u R3GAN i
pokazanie, że efekt nie wynika wyłącznie z większej pojemności modelu, innej procedury treningowej
albo innego budżetu obliczeniowego.
Jest to szczególnie ważne dlatego, że aktualny baseline nie jest oficjalnym kodem R3GAN, tylko
własną, spójną implementacją z własnym szkieletem generatora i dyskryminatora, relatywną funkcją
straty, zero-centered gradient penalty dla real i fake, EMA oraz nowoczesnym training loopem.
Roadmapa powinna więc rozszerzać ten kod modularnie, a nie podmieniać go na inny GAN tylko
dlatego, że tak było wygodniej w poprzedniej wersji.
Sam kierunek frequency-aware jest sensowny merytorycznie: problemy z wysokimi
częstotliwościami, strukturami drobnoskalowymi i jakością sygnału z dyskryminatora są realne, ale
sama obecność DWT nie jest już nowością. Wkład trzeba więc budować na sposobie integracji,
kontroli eksperymentalnej i jakości analizy, a nie wyłącznie na haśle użyłem waveletów.
1. Trzy sensowne opcje badawcze
Opcja A - rekomendowana
Wavelet-aware discriminator + lekka regularizacja statystyk HF po stronie generatora + mocne
eksperymenty kontrolne.
To najlepsza opcja, gdy chcesz zachować uczciwość względem baseline'u R3GAN, mieć realną
szansę na pozytywny wynik i dać temu formę sensownego rozdziału doktorskiego albo publikacji.
Wersja minimalna: tylko gałąź waveletowa w dyskryminatorze, tylko Haar, poziom 1, tylko pasma HF
(LH/HL/HH), fuzja po pierwszym downsamplingu w D, bez zmiany generatora i bez wavelet output.
Wersja pełna: Wavelet branch w D, generator-only wavelet statistic matching, kontrola FFT, kontrola
matched-capacity bez waveletów, dwa datasety i finalnie trzy seedy.
Opcja B - bardziej konserwatywna
Gałąź metodologiczna: diagnostyka częstotliwościowa + protokół stabilności i jakości, bez zmiany
architektury.
To ma sens, gdy chcesz najpierw zbudować porządne środowisko oceny, nie jesteś pewien, czy
sama modyfikacja modelu da wyraźny zysk, albo zależy Ci na materiale do rozdziału o metrykach i
diagnostyce GAN.
To słabsze jako nowa metoda, ale mocne metodologicznie.
Opcja C - wysoki risk / wysoki koszt
Wavelet-output generator albo learnable wavelets.
Ta opcja ma sens dopiero wtedy, gdy opcja A już działa, chcesz zxwrobić eksperyment rozszerzający i
jesteś gotowy na większe ryzyko interpretacyjne.
Problem polega na tym, że miesza zmianę reprezentacji wyjścia generatora, zmianę architektury,
zmianę przepływu gradientów i często także zmianę pojemności modelu. Trudniej ją obronić jako
uczciwe porównanie z R3GAN.
2. Co należy zmienić względem starej roadmapy
 Nie zmieniałbym baseline'u na inny ResNet GAN z hinge loss. W obecnym kontekście
metodologicznym nie ma to sensu - punktem odniesienia ma zostać Twój własny baseline
R3GAN.
 Nie dodawałbym regularizacji waveletowej do loss_D. Wavelet branch powinien być w
dyskryminatorze, ale wavelet statistic regularizer powinien trafiać wyłącznie do generatora.
 Nie zaczynałbym od db2 ani od poziomu L=2. Na start wyłącznie haar, level=1 i HF-only.
 Dodałbym koniecznie kontrolę matched-capacity. Bez niej zawsze będzie można powiedzieć, że
poprawa wynika z większej pojemności modelu, a nie z reprezentacji waveletowej.
 Porównania robiłbym po kimg albo images seen, a nie po samych iteracjach. Przy różnych batch
size porównanie po iteracjach jest metodologicznie słabe.
3. Rekomendowana struktura eksperymentu
Najpierw należy zamrozić baseline eksperymentalny R3GAN: ten sam loss, te same optymalizatory,
EMA, preprocessing, augmentacja, budżet treningu wyrażony w kimg, ta sama ewaluacja i te same
seedy dla porównań finalnych.
Następnie trzeba zbudować pełną otoczkę eksperymentalną: runner eksperymentów, spójne
logowanie, kontrolowaną ewaluację oraz metryki główne i pomocnicze.
Dopiero potem wdrażasz fixed Haar DWT, wavelet branch w dyskryminatorze, matched-capacity
control oraz generator-only wavelet statistic regularization. FFT regularizer powinien pełnić rolę
kontrolną, a nie zastępować główny pomysł.
Pełny eksperyment powinien być realizowany w trzech fazach: szybki test wykonalności,
eksperyment rozwojowy do wyboru najlepszego wariantu oraz eksperyment potwierdzający z kilkoma
seedami i więcej niż jednym ustawieniem danych.
4. Jak oceniać wynik
 Metryki główne: FID, KID, Precision / Recall oraz LPIPS-diversity jako pomocnicza miara
zróżnicowania percepcyjnego.
 Metryki pomocnicze: RPSE i WBED, ale traktowane jako diagnostyka mechanizmu, a nie jako
ostateczna miara jakości generacji.
 Za realną poprawę należy uznać wynik, który poprawia FID i KID, nie pogarsza wyraźnie
Precision / Recall, nie zapada LPIPS-diversity, poprawia miary HF zgodnie z hipotezą i utrzymuje
się po matched-capacity control.
 Dodatkowo warto raportować AUC krzywej FID-vs-kimg, success rate treningu, liczbę
parametrów, czas kroku i peak VRAM.
5. Ryzyka i pułapki interpretacyjne
 Model może wyglądać lepiej tylko dlatego, że ma większą pojemność - stąd konieczność control
branch o podobnym koszcie.
 Regularizacja HF może produkować sztuczny szum wysokoczęstotliwościowy zamiast realnej
poprawy - dlatego potrzebne są Precision / Recall, LPIPS-diversity, nearest-neighbor checks i
inspekcja wizualna.
 Gałąź waveletowa może destabilizować D - dlatego startujesz od HF-only, level 1, gate init = 0 i
zero-init ostatniej konwolucji.
 Łatwo pomylić ładniejszy wykres lossów z realną stabilnością treningu. Dlatego należy
raportować nie tylko lossy, ale także AUC FID-vs-kimg, success rate oraz wariancję między
seedami.
 Nie warto mieszać tej gałęzi z CLIP, DINO czy metrykami semantycznymi na etapie startowym.
To osobna oś badawcza.
6. Ocena innowacyjności
Ten kierunek może stanowić realny wkład naukowy, ale nie w wersji wrzuciłem DWT do GAN. Wkład
pojawia się dopiero wtedy, gdy integracja z R3GAN jest kontrolowana, efekt przechodzi przez
matched-capacity control, poprawa jest powtarzalna między seedami, nie kończy się na własnych
metrykach pomocniczych i koszt obliczeniowy jest jawnie raportowany.
Jeżeli poprawa jest marginalna, znika po control branch albo działa wyłącznie na jednym bardzo
wygodnym zbiorze danych, to będzie to raczej solidna inżynieria niż mocny wkład badawczy.
Najmocniejsza narracja polega na pokazaniu, że baseline R3GAN można rozszerzyć o kontrolowany
bias częstotliwościowy w sposób minimalistyczny, metodologicznie uczciwy i praktycznie użyteczny,
wraz z jasnym wskazaniem, kiedy to pomaga, a kiedy nie.
Część II. Nowa roadmapa implementacyjno-badawcza
Poniższa roadmapa jest już napisana pod Twój aktualny kontekst: własny baseline R3GAN, PyTorch,
MSI EdgeXpert z układem GB10 oraz nacisk na uczciwy protokół eksperymentalny i wdrożenie przez
mniej zaawansowanego wykonawcę.
Jest to roadmapa badawczo-implementacyjna, a nie szkic paperu pod szybki wynik.
0. Założenie główne
Ta gałąź badawcza ma odpowiadać na pytanie: czy można poprawić stabilność i lub jakość generacji
baseline'u R3GAN przez kontrolowane wprowadzenie biasu częstotliwościowego, bez rozwalania
uczciwości porównania?
Wersja minimalna to rozszerzenie tylko dyskryminatora o małą gałąź waveletową HF, bez zmiany
generatora. Wersja pełna obejmuje wavelet-aware discriminator, generator-only wavelet statistic
regularization, FFT control baseline oraz matched-capacity control.
Na starcie nie robimy wavelet-output generatora, learnable wavelets, poziomu L=2, db2, mieszania z
CLIP, DINO czy zmian bazowego lossu adversarialnego.
1. Architektura docelowego eksperymentu
Warianty porównawcze powinny być zdefiniowane jasno: R0 - Baseline, R1 - Matched-capacity
control, R2 - WaveD, R3 - WaveD + WaveReg, R4 - WaveD + FFTReg.
Taki zestaw umożliwia oddzielenie efektu dodatkowej gałęzi od efektu reprezentacji waveletowej i od
efektu samej regularizacji częstotliwościowej.
2. Co w obecnym kodzie zostaje bez zmian
Na starcie nie ruszasz generatora, logiki presetów, głównego lossu relatywistycznego, gamma, EMA,
Adama, channels-last ani AMP dla generatora. To są elementy baseline'u, które mają zostać
zamrożone na potrzeby tej gałęzi badań.
Najważniejsza decyzja konstrukcyjna to nie przepisywać całego dyskryminatora od zera, tylko zrobić
jego rozszerzenie albo opcjonalny branch.
3. Nowa struktura plików
Rekomendowany podział to: wavelets.py, wavelet_branches.py, freq_regularizers.py,
metrics_freq.py, trainer_wavelet.py, experiment_configs.py, run_experiment.py, analyze_results.py
oraz katalog tests z testami jednostkowymi i integracyjnymi.
Jeśli chcesz ograniczyć liczbę plików, minimalny układ to cztery moduły: wavelets.py,
freq_regularizers.py, trainer_wavelet.py i metrics_freq.py.
4. Nowe konfiguracje
Warto dodać osobne dataclasses dla WaveletConfig, FrequencyRegularizerConfig oraz
ExperimentControlConfig, aby eksperymenty były jawnie sterowane konfiguracyjnie.
Kluczowe pola to między innymi: wavelet_type='haar', level=1, hf_only=True, fuse_after_stage=0,
init_gate=0.0, lambda_wave, lambda_fft, use_log_energy, ema_stats_beta, eval_every_kimg i
total_kimg.
5. Etap 1 - DWT / IDWT
Najpierw budujesz poprawne, proste i testowalne wavelety. Na start tylko FixedHaarDWT2d i
FixedHaarIDWT2d.
Implementacja powinna używać stałych filtrów 2x2 zapisanych jako buffers oraz conv2d z
groups=in_channels. Na początek zakładasz parzyste H i W, bez paddingu, bez db2 i bez multi-level.
Dopiero gdy IDWT(DWT(x)) jest praktycznie idealne na losowym tensorze i na realnym batchu,
przechodzisz dalej.
6. Etap 2 - gałąź waveletowa do D
Celem jest dodanie częstotliwościowego sygnału do dyskryminatora bez przebudowy całej
architektury.
Najlepszy punkt fuzji to po pierwszym stage'u dyskryminatora. Obraz RGB trafia normalnie do
from_rgb, przechodzi przez pierwszy stage D, a równolegle trafia do DWT. Z pasm LH, HL i HH
budujesz mapę cech zgodną z wyjściem pierwszego stage'u i dodajesz ją przez residual fusion.
W branchu powinny znaleźć się dwie małe konwolucje, zero-init ostatniej konwolucji oraz uczony
gate z inicjalizacją 0.0. Dzięki temu model startuje dokładnie jak baseline i dodatkowa gałąź wchodzi
do gry dopiero wtedy, gdy trening uzna ją za użyteczną.
7. Etap 3 - refaktor dyskryminatora
Najczytelniejsza dla wykonawcy wersja to stworzenie nowej klasy, na przykład
WaveletR3GANDiscriminator, zamiast agresywnie modyfikować starą klasę w miejscu.
Forward powinien zachować tę samą logikę bazową: odczyt RGB, from_rgb, pierwszy stage, dodanie
aux_branch, reszta stages i opcjonalna projekcja warunkowa. Ważne jest, aby wavelet branch
widział dokładnie to samo wejście po augmentacji, które trafia do D.
8. Etap 4 - matched-capacity control
Ten eksperyment jest obowiązkowy. Branch kontrolny powinien mieć bardzo podobną strukturę,
koszt i liczbę parametrów, ale nie korzystać z DWT. Najprościej użyć avg_pool2d na RGB, a potem
tej samej małej wieży konwolucyjnej i tego samego gate'a.
Ten wariant odpowiada na zarzut, że poprawa bierze się z dodatkowej ścieżki, a nie z reprezentacji
waveletowej.
9. Etap 5 - refaktor trainera
Obecny train_step należy podzielić na _discriminator_step, _generator_step oraz train_step jako
wrapper. Jest to potrzebne, aby w kroku G policzyć jednocześnie adversarial loss i regularizer
częstotliwościowy.
W kroku generatora powinieneś mieć fake = G(z, cond), dalej fake_logits i real_logits, obliczenie
g_adv, obliczenie g_reg i finalnie g_total = g_adv + g_reg.
WaveReg i FFTReg dodajesz wyłącznie do generatora, nie do loss_D.
10. Etap 6 - Wavelet statistic regularizer
Regularizer ma dopasowywać rozkład energii wysokich częstotliwości bez paired loss. Dla każdego
obrazu wykonujesz DWT, bierzesz LH, HL i HH, liczysz mean(coeff^2), najlepiej przechodząc potem
na log-energy przez log(E + eps).
Po batchu liczysz mu_fake, std_fake oraz odpowiadające im EMA statystyki realnych obrazów.
Strata może mieć postać lambda * (L1(mu_fake, mu_real_ema) + 0.5 * L1(std_fake, std_real_ema)).
Warto logować nie tylko całość, ale także poszczególne składowe i średnie energii dla każdego
pasma.
11. Etap 7 - FFT control regularizer
Ten moduł powinien być możliwie równoległy do waveletowego: FFT2, moc widmowa, radial binning,
pominięcie DC, 16 binów, mean i std po batchu oraz analogiczna funkcja straty.
Dzięki temu porównujesz, czy pomaga sam prior częstotliwościowy, czy konkretnie waveletowy
sposób reprezentacji.
12. Plan eksperymentów
Faza A: infrastruktura i sanity - zamrożenie baseline'u, runner, logging, metryki, testy waveletów. Nie
przechodzisz dalej, dopóki baseline nie działa stabilnie, metryki liczą się poprawnie, a DWT i IDWT
przechodzą testy.
Faza B: szybki test sensu naukowego - porównanie baseline, matched-capacity i WaveD na jednym
datasecie, jednej rozdzielczości i jednym seedzie.
Faza C: pełna metoda - baseline, matched-capacity, WaveD, WaveD + WaveReg, WaveD + FFTReg
na jednym datasecie i jednej rozdzielczości, ale już na dwóch seedach, aby wybrać finalny wariant.
Faza D: eksperyment potwierdzający - baseline kontra najlepszy wariant, dwa datasety albo jeden
dataset i dwie rozdzielczości, finalnie trzy seedy. Na początek lepiej postawić na dwa datasety przy
tej samej rozdzielczości niż na skakanie po wielu rozdzielczościach.
13. Krytyczne ustawienia i hiperparametry startowe
W obrębie porównania identyczne muszą być: preset, trainer config, gamma, EMA, optimizer,
preprocessing, augmentacja, batch size, budżet w kimg, harmonogram ewaluacji, liczba próbek do
FID i KID, seedy i latent vectors do sample grids.
Jako punkt wejścia zostawiasz obecne ustawienia baseline'u: lr_g = 2e-4, lr_d = 2e-4, betas = (0.0,
0.99), gamma = 10.0, ema_beta = 0.999, use_amp_for_g = True, use_amp_for_d = False,
amp_dtype = bfloat16, channels_last = True.
Dla WaveD startujesz od haar, level=1, hf_only=True, fuse_after_stage=0, gate=0.0,
branch_mid_scale=0.5. Dla WaveReg: lambda_wave = 0.02 jako start oraz sweep 0.005, 0.02, 0.05.
Dla FFTReg: lambda_fft = 0.02 i 16 radial bins.
14. Jak oceniać wynik i kiedy uznać go za realny
Metryki główne to FID, KID, Precision / Recall oraz LPIPS-diversity. Pomocniczo RPSE i WBED.
Dodatkowo warto raportować liczbę parametrów, czas kroku, peak VRAM, AUC FID-vs-kimg i
success rate treningu.
Realna poprawa jest wtedy, gdy poprawiają się FID i KID, nie pogarsza się wyraźnie Precision /
Recall, LPIPS-diversity nie zapada się, RPSE i WBED poprawiają się zgodnie z hipotezą, matched-
capacity control nie robi tego samego, a wynik utrzymuje się między seedami.
Jeżeli poprawiają się tylko RPSE i WBED, a FID i KID stoją, to nie masz jeszcze poprawy modelu -
raczej poprawę zgodności częstotliwościowej.
15. Checklist dla wykonawcy i szkic do rozprawy
Kolejność wdrożenia powinna być sztywna: baseline i metryki, DWT i testy, WaveD, matched-
capacity, WaveReg, FFT control, a dopiero potem ablacjki.
Zadania implementacyjne można rozbić na kolejne taski: dodanie wavelets.py, testów rekonstrukcji,
branchy pomocniczych, nowej klasy dyskryminatora, regularizerów, refaktoru trainera, metryk
częstotliwościowych, runnera eksperymentów i skryptu do analizy wyników.
Do rozprawy możesz opisać tę gałąź jako badanie, które sprawdza, czy kontrolowane rozszerzenie
baseline'u R3GAN o pomocniczy sygnał częstotliwościowy poprawia jakość i stabilność generacji
bez zmiany podstawowego celu adversarialnego. Hipoteza, opis eksperymentu, interpretacja oraz
ograniczenia wynikają bezpośrednio z powyższego protokołu.
Podsumowanie praktyczne
Werdykt: ta roadmapa ma sens i jest znacznie lepiej dopasowana do Twojego doktoratu niż
poprzednia wersja, bo nie zmienia baseline'u, jest modularna, ma uczciwe kontrole, rozdziela efekt
architektury od efektu biasu częstotliwościowego i jest realnie wdrażalna na Twoim kodzie.
Najważniejsze jest zachowanie kolejności prac oraz nierozszerzanie zakresu zbyt wcześnie.