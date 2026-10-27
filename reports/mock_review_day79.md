# Mock Review (Day 79)

1. Problem (30 sec)    : "Retailers lose money when customers silently stop buying.
                          I built a model that ranks customers by churn risk so a limited
                          retention budget goes to the right people."
2. Data (30 sec)       : "~1M transactions, UK online retailer, 2009-2011.
                          I cleaned it: [X]% rows had no Customer ID, cancellations removed."
3. Target design (60)  : "My first churn label used Recency over the whole period, which leaks.
                          I redesigned it: features from before a cutoff, label from the
                          90 days after."     <- yeh tumhari sabse strong kahani hai
4. Analysis (45 sec)   : "EDA and hypothesis tests showed [one real finding with number]."
5. Modeling (60 sec)   : "Compared LogReg and a tuned tree with 5-fold CV, handled imbalance,
                          confirmed once on a held-out test set."
6. Result + honesty (45): "Test ROC-AUC [X], top-10% targeting catches [X]% of churners.
                          Limitation: [ek sachchi limitation]."
7. Next (30 sec)       : "Next I'm adding ensembles, SHAP and a Streamlit app."




## Must-fix (interviewer pakad lega)
- Har sawaal ke neeche likha hai ki ek achha jawab kya cover karta hai.

Data aur target

1. “Customer ID missing rows kyun drop ki?” Bina ID ke customer-level analysis possible nahi, ye guest checkouts hain. Number batao kitni % thi, aur ye bhi maano ki ye ek bias laa sakta hai (guest buyers ka behaviour alag ho sakta hai).
2. “Churn kaise define kiya? Aur 90 din kyun?” Business assumption hai, universal rule nahi. Dusra threshold lagane par churn rate badlega. Day 35B mein 60/75/90 ka sensitivity check yaad karo.
3. “Data leakage kya hai aur tumhare project mein kahan tha?” Purana label aur Recency dono ek hi window se bane the, model sirf rule ratta. Fix: observation window aur prediction window alag. Ye sabse zyada poochha jaane wala sawaal hoga.
4. “Tumhare EDA mein Recency aur churn ka relation ‘obvious’ kyun tha?” Purane label mein churn ki definition hi Recency thi (circular).

Statistics
5. “p-value ki sahi definition?” H0 sach hone par itna extreme data aane ki probability. H0 ke sach hone ki probability nahi.
6. “Bade sample mein sab kuch significant kyun aa jaata hai? Tumne kya kiya?” Effect size (Cohen’s d, Cramér’s V) saath mein dekha.
7. “Monetary skewed hai, t-test par kitna bharosa?” Log-transform aur Mann-Whitney se cross-check kiya.

Modeling
8. “Scaling pipeline ke andar kyun?” Taaki har CV fold mein scaler sirf train hisse par fit ho, validation par leak na ho.
9. “SMOTE test set par kyun nahi lagta?” Test set asli duniya jaisa rehna chahiye. Synthetic points validation ya test mein aaye to score jhootha ban jaata hai.
10. “class_weight, SMOTE aur baseline mein se kya chuna aur kyun?” Apna Day 64C table dekh kar jawab do. “Maine SMOTE use kiya kyunki popular hai” kamzor jawab hai.
11. “Tumne tuning kaise ki, aur test set kab khola?” CV par tuning, test sirf confirmation ke liye ek baar.

Metrics
12. “Accuracy kyun nahi?” Naive baseline ka number batao (majority-class accuracy).
13. “Precision aur Recall mein kya chunoge?” Business cost ke hisaab se: campaign sasti ho to recall, mehengi ho to precision. Apne assumed costs ka reference do.
14. “AUC 0.5 aur 0.8 ka matlab bina jargon ke?” Ek churner aur ek active customer randomly chuno, model 80% baar churner ko zyada risky rank karta hai.
15. “Agar AUC 0.98 aata to?” Pehle leakage ka shak karta.

Mushkil follow-ups
16. “Agar main kal naya data deta, to model kab tak chalta?” Drift hota hai, retraining ki zaroorat padegi (Month 5 mein monitoring aayegi).
17. “Probabilities calibrated hain?” Nahi, class weighting se inflate hoti hain, isliye model ranking ke liye hai.
18. “Is project ki sabse badi kamzori kya hai?” Ek sachchi kamzori batao (jaise sirf ek retailer ka data, test mein churners kam).


Ek problem jo tumhare repo mein hone ki bahut sambhavna hai: do alag churn definitions. Purani wali (Day 35B) poore period par bani thi, aur nayi (Day 62) future window par. EDA notebooks mein churn rate ek aur ML section mein alag hoga. Agar README mein dono ka fark samjhaya nahi, to interviewer ko lagega tum numbers galat likh rahe ho. Check karo:


Customers ki sankhya alag aa sakti hai, aur wo theek hai. ML dataset sirf unhe rakhta hai jinhone cutoff se pehle kuch kharida tha. Ye difference README ke “Target Variable” section mein ek line mein likhna zaroori hai:

“EDA uses a descriptive Recency-based label over the full period. The ML models use a separate leakage-free label (features before the cutoff, outcome in the 90 days after), so the two churn rates differ by design.”

Ab har number ko README se piche trace karo. Teen numbers chuno (churn rate, best model ka test F1, precision@10%) aur har ek ke liye likho: README ka number → kaunsa notebook → kaunsa cell → kaunsi file. Agar koi number tum trace nahi kar paye, to wo number galat ho sakta hai ya README mein guess likha hai.

## Should-fix (professional lagta hai)
- [ ] ...

## Nice-to-have
- [ ] ...

## Sawaal jin par atka
1. ...

## Pitch ke notes
- Kaatna hai: ...
- Yaad karna hai (numbers): ...