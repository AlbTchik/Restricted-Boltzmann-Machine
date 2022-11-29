# Machine de Boltzman Restreinte

**Date de réalisation :** Octobre 2022, Cours de Raisonnement & Modèles Probabilistes  
<br>
**Cadre du projet :** Génération de chiffres avec une RBM  
<br>
**Mots-clés :** Machine de Boltzmann, Génération d'images, Dataset MINST   

## Présentation du projet

Voici la fameuse machine de Blotzmann, l'ancêtre des réseaux de neurones artificels. Initialement inventée sous le nom de Harmonium par Paul Smolenski en 1986.



## Partie Théorique

### 1) Structures des RBM

On va construire une Machine de Boltzmann Restreinte (RBM) pour générer une représentation améliorée des chiffres.
Les RBM comportent des unités cachées et visibles et consiste en une matrice de poids $W$ de taille $m × n$.

Chaque élément de poids $w_{i,j})$ de la matrice est associé à la connexion entre l'unité visible (d'entrée) $x_{i}$ et l'unité cachée $h_{j}$. En outre, il existe des poids de biais (décalages) $a_{i}$ pour $x_{i}$ et $b_{j}$ pour $h_{j}$, cela est illustré dans le figure suivante : 

![RBM](https://user-images.githubusercontent.com/90097422/204649320-cd61c151-bb3c-447d-918c-5903edb762a3.png)

Compte tenu des poids et des biais, l'énergie d'une configuration (paire de vecteurs booléens) (x, h) est définie comme suit :  

$$E(x ,h) = -h^T W x -a^T x -b^T h = -\sum_{j}\sum_{k} (w_{j,k} h_j x_k) -\sum_{k} (a_k x_k) -\sum_{j} (b_j h_j)$$


La distribution de probabilité conjointe pour les vecteurs visibles et cachés est définie en termes de fonction d'énergie comme suit :  

$$p(x, h) = \frac{e^{-E(x, h)}}{Z_θ} = \frac{e^{h^T W x +a^T x +b^T h}}{Z_θ} = \frac{e^{(h^T W x} + e^{a^T x} + e^{b^T h}}{Z_θ} = \frac{\prod_{j}\prod_{k} (w_{j,k} h_j x_k) * \prod_{k} (a_k x_k) * \prod_{j} (b_j h_j)}{Z_θ}$$

où $Z_θ$ est une fonction de partition définie comme la somme de $e^{-E(x, h)}$ sur toutes les configurations possibles, qui peut être interprétée comme une constante de normalisation pour garantir que la somme des probabilités est égale à 1.  

La probabilité marginale d'un vecteur visible est la somme de $P(x, h)$ sur toutes les configurations possibles de la couche cachée :  

$$P(x)=\frac{1}{Z_θ}\sum _{h} e^{-E(x, h)}$$
    ,

et vice versa. Comme la structure graphique sous-jacente du RBM est bipartite (ce qui signifie qu'il n'y a pas de connexions intra-couche), les activations des unités cachées sont mutuellement indépendantes des activations des unités visibles.  

Inversement, les activations des unités visibles sont mutuellement indépendantes des activations des unités cachées, c'est-à-dire que pour m unités visibles et n unités cachées, la probabilité conditionnelle d'une configuration des unités visibles x, compte tenu d'une configuration des unités cachées h, est la suivante : 

$$P(x|h) = \prod_{i=1}^{m} P(x_i|h)$$
    .

Inversement, la probabilité conditionnelle de h étant donné v est : 

$$P(h|x) = \prod_{j=1}^{n} P(h_j|x)$$.

Les probabilités d'activation individuelles sont données par

   $$P(h_{j}=1|x)=\sigma \left(b_{j}+\sum _{i=1}^{m}w_{i, j}x_{i}\right)$$ 
   $$P(x_i=1|h) = \sigma \left(a_i + \sum_{j=1}^n w_{i,j} h_j \right)$$

où σ désigne la sigmoïde logistique.

### 2) Entrainement des RBM

Les machines de Boltzmann restreintes sont entraînées pour maximiser le produit des probabilités attribuées à un certain ensemble d'entraînement $X$ (une matrice, dont chaque ligne est traitée comme un vecteur visible $x$),

$$\arg\max_W \prod_{x \in X} P(x)$$

L'algorithme le plus souvent utilisé pour entraîner les RBM, c'est-à-dire pour optimiser la matrice de poids $W$, est l'algorithme de divergence contrastive d'Hinton. L'algorithme effectue un échantillonnage de Gibbs et est utilisé à l'intérieur d'une procédure de descente de gradient pour calculer la mise à jour des poids.

La procédure de base de divergence contrastive en une seule étape pour un seul échantillon peut être résumée comme suit :

1. Prenez un échantillon de formation x, calculez les probabilités des unités cachées et échantillonnez un vecteur d'activation caché h à partir de cette distribution de probabilité.

2. Calculer le produit externe de v et de h et l'appeler le gradient positif.

3. À partir de h, échantillonnez une reconstruction x' des unités visibles, puis rééchantillonnez les activations cachées h' à partir de celle-ci. (étape d'échantillonnage de Gibbs)

4. Calculer le produit externe de x' et h' et l'appeler le gradient négatif.

5. La mise à jour de la matrice de poids $W$ est le gradient positif moins le gradient négatif, multiplié par un certain taux d'apprentissage : $\Delta W=\epsilon (vh^{\mathsf {T}}-v'h'^{\mathsf {T}})$.
    
6. Mettre à jour les biais a et b de manière analogue : $\Delta a=\epsilon (v-v')$, $\Delta b=\epsilon (h-h')$.

Finalement, on cherche a atteindre un minimum de la fonction de perte par des mise a jour successives, comme présenté dans la figure ci-dessous :

![bassin of attraction](https://user-images.githubusercontent.com/90097422/204649403-430a4607-308d-4683-967d-a1009eb3683b.png)

