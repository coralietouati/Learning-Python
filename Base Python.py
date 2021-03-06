
################################## LES CONTENEURS: LISTES, DICO ... #################################################################
# Operations
    print(10/3)
    print(10.0/3)
    print(10.0//3)
    print(10%3)
    print(2,5+1)
    print(2.5+1) # toujours utiliser un point
    print(2!=3)
    print(2==3)

# String
#Les méthodes de la classe str, sans exception, travaillent sur la chaîne et renvoient la chaîne modifiée, sans modifier la chaîne d'origine

chaine = str()
chaine = "TEST"
chaine = chaine.lower() # put in lower cases
chaine.upper() # put in upper cases
chaine = chaine.capitalize() #first letter with upper case"
chaine = " TEST  2 "
chaine.strip() # delete first and last spaces if any

# selection de caractere dans une chaine
message = "bonjour"
message[0:2] # deux premieres lettres
message[3:len(message)] # tout sauf les deux premieres


# PRINT options
name = " Martine"
age = 21
sentence ="It is {0}'s Birthday she is {1}".format(name,age)

# formatage d'une adresse
adresse = """
{no_rue}, {nom_rue}
 {code_postal} {nom_ville} ({pays})
""".format(no_rue=5, nom_rue="rue des Postes", code_postal=75003, nom_ville="Paris", pays="France")
print(adresse)


#### LES LISTES
list  = [] # create an empty list
list.append(1) # add   /!\ objet d'orgine est directement modifier avec les listes
list.append(2)
list.inset(0,100) # ajouter une valeur avec un index precis
list.insert(1,'a')
list2 = [4,6,8]
list3 =list + list2
list.extend(list2) # directly modify list
list += list2 # same as extend
del list[0]
list.remove('a')  # La méthode remove ne retire que la première occurrence de la valeur trouvée dans la liste !

# Trick to show the elements of the list
for elt in list:
    print(elt)
for i, elt in enumerate(list):
    print("element at index {0} is {1}".format(i,elt))
for elt in enumerate(list):
    print(elt)

# Liste <=> Chaine
    chaine = 'Bonjour Martine !'
    list = chaine.split() # transform the string in a list
    " ".join(list) # transform the list in chain

# Trier
    prenoms = ["Jacques", "Laure", "Andre", "Victoire", "Albert", "Sophie"]
    prenoms.sort() # modifie l'object et tri par ordre alphabetique
    # seul methode qui modifie l4indice de la liste
    sorted(prenoms) # renvoie un nouvel objet

# Fonctions et lites
l=[1,2,3]
def f(l1,l2,l3):
    print(l1)
f(*l)

    # Les compréhensions de liste (« list comprehensions » en anglais) sont un moyen de filtrer ou modifier une liste très simplement.
    [nb * nb for nb in liste]
    [nb for nb in liste if nb % 2 == 0]

    # Gestion stock de fruit
    qtt_a_retirer = 7  # On retire chaque semaine 7 fruits de chaque sorte
    fruits_stockes = [15, 3, 18, 21]  # Par exemple 15 pommes, 3 melons...
    fruits_stockes = [nb - qtt_a_retirer for nb in fruits_stockes if nb > qtt_a_retirer]

    inventaire = [("pommes", 22), ("melons", 4), ("poires", 18), ("fraises", 76), ("prunes", 51)]
    inventaire.sort()  # tri sur le nom des fruits et non sur la quantite
    print(inventaire)

    inventaire_invers = [(qtt, nom) for (nom, qtt) in inventaire]  # inverser les tuplets
    inventaire_invers.sort()  # tri sur la qtt
    print(inventaire_invers)

    inventaire_trie = [(nom, qtt) for (qtt, nom) in inventaire_invers]

    ##### Les DICOTIONAIRES
    mon_dictionnaire = {}
    mon_dictionnaire["pseudo"] = "Prolixe"
    mon_dictionnaire["mot de passe"] = "*"

    echiquier = {}
    echiquier['a', 1] = "tour blanche"  # En bas à gauche de l'échiquier
    echiquier['b', 1] = "cavalier blanc"  # À droite de la tour

    placard = {"chemise": 3, "pantalon": 6, "tee-shirt": 7}

    # Supprimer des clés d'un dictionnaire
    placard = {"chemise": 3, "pantalon": 6, "tee shirt": 7}
    del placard["chemise"]  # mot cle del
    placard = {"chemise": 3, "pantalon": 6, "tee shirt": 7}
    placard.pop("chemise")  # methode pop, affiche la valeur supprimee

    # parcourir les valeurs d'un dico
    fruits = {"pommes": 21, "melons": 3, "poires": 31}
    for cle in fruits.keys(): # renvoie la liste des fruits
        print(cle)
    if 21 in fruits.values(): # liste des chiffres
        print("Un des fruits se trouve dans la quantité 21.")
    for cle, valeur in fruits.items(): # items renvoie le couple cle,valeur
        print("La clé {} contient la valeur {}.".format(cle, valeur))

    # stocker des fonctions
    def fete():
        print('cest la fete')
    def oiseau():
        print("fais comme l'oiseau")
    fonctions = {}
    fonctions['go'] = fete
    fonctions['cuicui'] = oiseau
    fonctions['go']()

    #On peut capturer les paramètres nommés passés à une fonction en utilisant cette syntaxe :def fonction_inconnue(**parametres_nommes)
    print("Voici", "un", "exemple", "d'appel", sep=" >> ", end=" -\n")
    # on peut ecrire la meme chose avec un dico
    parametres = {"sep": " >> ", "end": " -\n"}
    print("Voici", "un", "exemple", "d'appel", ** parametres)

    #TRI
    etudiants = [
        ("Clement", 14, 16),
        ("Charles", 12, 15),
        ("Oriane", 14, 18),
        ("Thomas", 11, 12),
        ("Damien", 12, 15), ]
    # tir avec fct landa 5plus lent)
    sorted(etudiants,key= lambda column : column[2])
    # plus rapide avec l'opertor
    from operator import itemgetter
    sorted(etudiants, key= itemgetter(2))


    # Remplacer un dico par une class
class Etudiant:

    def __init__(self, prenom, age, moyenne):
        self.prenom = prenom
        self.age = age
        self.moyenne = moyenne

    def __repr__(self):
        return "Etudiant {} age= {} moyenne = {}".format(self.prenom,self.age,self.moyenne)

etudiants = [
    Etudiant("Clément", 14, 16),
    Etudiant("Charles", 12, 15),
    Etudiant("Oriane", 14, 18),
    Etudiant("Thomas", 11, 12),
    Etudiant("Damien", 12, 15),
]

print(etudiants[1])
sorted(etudiants, key = lambda etudiant : etudiant.moyenne, reverse=True) # la variable est l'objet et renvoie l'attribut
from operator import attrgetter
sorted(etudiants, key= attrgetter('moyenne','age'))
# astuce: pour trier selon deux critère opposés, tri selon le deuxieme puis le premier --> garde l'ordre grace prorpirete de stabiite


    ### SET
    # Unset(ensemble) est un objet conteneur (lui aussi), très semblable aux listes sauf qu'il ne peut contenir deux ' \
    # objets identiques. Vous ne pouvez pas trouver deux fois dans unsetl'entier3par exemple.
    set = {'cle1', 'cle2'}

    ################################### LOOP #########################################################################
# Types de variables
    chaine = "J'aime le \n Python"
    print(chaine)

# Astuces
    y = x = 2
    print("x=",x," y=",y)
    x += 1 # equivaut à x = x +1
    print("x=",x," y=",y)


# Boucle for
    chaine = "Hello"
    for letter in chaine :
        print(letter)
    print("")

    for i in range(0,5):
        print(i)

    x=[0,11,22,33,44]
    for indice,valeur in enumerate(x):
        print(indice,'val=',valeur)

    for indval in enumerate(x):
        print(indval)
        print('liste[', indval[0], '] = ', indval[1])
        print ("liste[%d] = %r" % indval)

# Boucle While
    nb = 7
    i = 0
    while i < 10:
        print(i + 1, "*", nb, "=", (i + 1) * nb)
        i += 1

# Boucle IF

# Ex 1
    #Ex1
    if 2 == 1 + 1 and 4 == 2 + 2:
        print("ok")
    else:
        print("pb")

    # Ex 2
    a = 1
    if a > 0:  # Positif
        print("a est positif.")
    elif a < 0:  # Négatif
        print("a est négatif.")
    else:  # Nul
        print("a est nul.")




####################################### FUNCTION #######################################################################

# Creer sa fonction
    def mySum(x,y=10,z=0):
        """Description de la fonction"""
        return x+y+z
    print(mySum(2)) # par défault y vaut 10
    print(mySum(2,2))
    print(mySum(x=2,z=2))
    ?mySum()

    # 2eme technique
    f = lambda x, y: x+y
    f(1,2)

## fonction inconnue
    # On peut découper une chaîne en fonction d'un séparateur en utilisant la méthodesplitde la chaîne.
    # On peut joindre une liste contenant des chaînes de caractères en utilisant la méthode de chaînejoin. Cette méthode doit être appelée sur le séparateur.

    # FOnction parametres inconnus: On peut créer des fonctions attendant un nombre inconnu de paramètres grâce
    # à la syntaxedef fonction_inconnue(*parametres):(les paramètres passés se retrouvent dans le tuple parametres).

    def fonction_inconnue(*parametres):
        """Test d'une fonction pouvant être appelée avec un nombre variable de paramètres"""
        print("j'ai recu {}".format(parametres))


    fonction_inconnue()
    fonction_inconnue(1, 'pomme')

    # print est aussi une fonction qui peut prendre un nb de parametre inconnu
    liste = [1, 2, 3, 4, 5]
    print(liste)
# Variables

x,y = 1,2




# EXCEPT AND TRY

numerateur = 6
denominateur = 0
try:
    resultat = numerateur / denominateur
except NameError:
    print("La variable numerateur ou denominateur n'a pas été définie.")
except TypeError:
    print("La variable numerateur ou denominateur possède un type incompatible avec la division.")
except ZeroDivisionError:
    print("La variable denominateur est égale à 0.")
else:
    print(resultat)  # or return si cest une fonction
finally:
    print(" This is done even if there is an other error found by python")

    # CAPTURER l'erreur
try:
    resultat = numerateur / denominateur
except Exception as x:
    print("Voici l'erreur :", x)


################################## LES REFERENCES #################################
L1 = [1, 2, 3]
L2 = L1
L2.append(4)
print(L1)
L1 is L2 # on compare les ref
# L1 a egalement ete modifiee !! L1 et L2 contiennent une référence vers le même objet :
# si on modifie l'objet depuis une des deux variables, le changement sera visible depuis les deux variables.
# ca ne marche pas sur les entiers float etc Les entiers, les flottants, les chaînes de caractères, n'ont aucune
# méthode travaillant sur l'objet lui-même. Les chaînes de caractères, comme nous l'avons vu, ne modifient pas l'objet
# appelant mais renvoient un nouvel objet modifié.

L1 = [1, 2, 3]
L2 = list(L1)
# Ici on crée un nouvel objet basé sur L1. Du coup, les deux variables ne contiennent plus la même référence : elles modifient des objets différents
#L2.append(4)
print(L1)
L1 is L2 # on compare les ref
L1 == L2 # on compare les contenu


# Variables globales
i=0
def inc_i():
     """Fonction chargée d'incrémenter i de 1"""
     global i # Python recherche i en dehors de l'espace local de la fonction
     i += 1
inc_i()

################################### ASTUCES COMPLEXES  #############################################################
# Incrementer le nom des variables
for i in range(1,2):
    exec("df" + str(i) + "='" + str(1) + "'")
print(df1)

# La somme d4une colonne avec filtre
Total_bat_input = df[(df['storage (kW)'] > 0)]['storage (kW)'].sum()

############################################   LES CLASSES   ######################################################

# Methode pour tous les objets de la classe
def __setattr__(self, nom_attr,val_attr):
    """Méthode appelée quand on fait objet.nom_attr = val_attr.
                On se charge d'enregistrer l'objet"""
    print('on a changer la methode de la classe mere objet')
    object.__setattr__(self,nom_attr,val_attr)
    self.enregistrer()

def __delattr__(self, nom_attr):
    """On ne peut supprimer d'attribut, on lève l'exception AttributeError
   pour supprimer un attribut, n'utilisez pas dans votre méthode del self.attribut. mais object.__delattr__"""

    raise AttributeError("Vous ne pouvez supprimer aucun attribut de cette classe")

class Personne:
    """Definit une classe Personne avec son nom, age etc"""

    def __init__(self, nom, prenom):
        """ Constructeur de notre classe, self est toujours en premier puis on ajoute les arguments de la classe
        Ici on cree des attributs de l'objet moi"""
        self.nom = nom
        self.prenom = prenom
        self.age = 33
        self._ville = 'nice'  # pas oublier le _ : convention pour dire que l'attribut est prive ( on ne peut pas y acceder hors de la classe), quand on appelle personne.ville ca ne marche pas

    def _get_ville(self):
        """Methode pour lire l'attribut ville ASSESSEUR"""
        #print('Access a lattribut ville')
        return self._ville

    def _set_ville(self, newville):
        """methode pour modifier la ville MUTATEUR"""
        print('DEMENAGEMENT')
        print("attention demenagement a {}".format(newville))
        self._ville = newville

    def __repr__(self):
        """Methode d'affichage d'objet (sans print) """
        return "nom ({}) prenom ({}) ville ({})".format(self.nom, self.prenom, self.ville)

    def __str__(self):
        """Methode permettant de personaliser le renvoie de print(objet)"""
        return "{} vient de {}".format(self.prenom, self.ville)

    # def __getattr__(self, item):
    #     """Methode: si l'attribut demande n'existe pas, on renvoie une alerte"""
    #     print("Attention, il n'y a pas de {} ici!".format(item))


    # def __setattr__(self, nom_attr, val_attr):
    #     """Méthode appelée quand on fait objet.nom_attr = val_attr.
    #     On se charge d'enregistrer l'objet"""
    #
    #     object.__setattr__(self, nom_attr, val_attr)
    #     self.enregistrer()

    ville = property(_get_ville, _set_ville) # sans cette ligne personne.ville ne renvoie rien!
    # si on definit que l'assesseur alors l'attribut ne peut etre modifie


moi = Personne('Titi', 'Coralie') # moi est une instance de la classe Personne

moi.nom
moi.prenom
moi.age
moi.age=25
moi.age
moi.ville
moi.ville='sf'


class Compteur:

    """Cette classe possède un attribut de classe qui s'incrémente à chaque fois que l'on crée un objet de ce type.
    Les attribut de classe ne sont pas en self. mais en nomdelaclassse."""

    objets_crees = 0

    def __init__(self):
        """a chaque fois quon cree un objet on incremente le cpmteur"""
        Compteur.objets_crees += 1

    def combien(cls):
        """ce nest pas une methode d'instance mais une methode de classe --> argument nest pas self mais cls"""
        print("Jusqu'a present {} objets ont ete crees".format(cls.objets_crees))
    combien = classmethod(combien) # permet de convertir la methode en methode de classe

    def start():
        """ Methode statique, qui donne la meme chose pour tous les objets, independant des attributs de l'objet et de la classe"""
        print('Comptons!')
    start = staticmethod(start)

print(Compteur.objets_crees)
a = Compteur() # ne pas oublier les parentheses quand la class na pas dargument
Compteur.combien()
a.combien()
a.start()
Compteur.start()
print(Compteur.objets_crees)
b = Compteur()
print(Compteur.objets_crees)
dir(a)

class TableauNoir:
    """Classe définissant une surface sur laquelle on peut écrire,
    que l'on peut lire et effacer, par jeu de méthodes. L'attribut modifié
    est 'surface"""

    def __init__(self):
        self.surface = ""

    def ecrire(self,message):
        """Méthode permettant d'écrire sur la surface du tableau.
        Si la surface n'est pas vide, on saute une ligne avant de rajouter
        le message à écrire"""
        if self.surface != "":
            self.surface = '\n'
        self.surface = message

    def lire(self):
        """Cette méthode se charge d'afficher, grâce à print, la surface du tableau"""
        print(self.surface)

    def effacer(self):
        self.surface = ""



t = TableauNoir()
t.ecrire("hello")
t.lire()
t.effacer()
t.lire()

#Les attributs de l'objet sont propres à l'objet créé : si vous créez plusieurs tableaux noirs, ils ne vont pas tous
# avoir la même surface. Donc les attributs sont contenus dans l'objet.

#Les méthodes sont contenues dans la classe qui définit notre objet. Quand vous tapeztab.ecrire(…),
#  Python cherche la méthode ecrire non pas dans l'objet tab, mais dans la classeTableauNoir.
#tab.ecrire(…), equivaut a TableauNoir.ecrire(tab, …). self est l'objet qui appelle la méthode.

# La fonction dir renvoie liste comprenant le nom des attributs et méthodes de l'objet qu'on lui passe en paramètre
dir(t)
t.__doc__
# Atribut special: dico qui contient les noms des attributs (cles) les valeurs des attributs (valeur)
t.__dict__


class listing:

    def __init__(self):
        self._liste = [1,3,6]

    def __getitem__(self, index):
        return self._liste[index]

    def __setitem__(self, index, value):
        self._liste = self._liste.append(value)

    def __contains__(self, item):
        if item in self._liste:
            return True
        else:
            return False

    def __len__(self):
        return len(self._liste)

L = listing()
2 in L
len(L)


class Duree:
    """Classe contenant des durées sous la forme d'un nombre de minutes
    et de secondes"""

    def __init__(self, min=0, sec=0):
        """Constructeur de la classe"""
        self.min = min  # Nombre de minutes
        self.sec = sec  # Nombre de secondes

    def __str__(self):
        """Affichage un peu plus joli de nos objets"""
        return "{0:02}:{1:02}".format(self.min, self.sec)

    def __add__(self, new):
        """Additioner une durée: object1 + duree en sec"""
        newduree = Duree() # il faut constuire l'object pour avoir l'attibut min et sec!
        newduree.min = self.min
        newduree.sec = self.sec
        newduree.sec += new
        if newduree.sec >= 60 :
            newduree.sec = newduree.sec % 60
            newduree.min += newduree.sec // 60
        return  newduree

    def __radd__(self, new):
        """ methode special dans l'auttre sens, on ajout un r !
        On renvoie add dans l'autre sens (ici laddition marche dans les deux sens)"""
        return self + new

    def __iadd__(self, new):
        """methode speciale pour surchar l'operateur +=
        Dans ce cas on travaille directement sur self"""
        self.sec += new
        if self.sec >60:
            self.sec = self.sec % 60
            self.min += self.min // 60
        return self

    def __eq__(self, autreduree):
        """" methode = compare les minutes et sec """
        if self.min == autreduree.min and self.sec == autreduree.sec:
            return True
        else:
            return False





duree= Duree(1,30)
duree2= Duree(1,30)
print(duree)
print(duree + 10) # appelle __ad__
print(70 + duree) # appelle __radd__
duree += 20 # appelle __iadd__
print(duree)


class Temp:
    """Classe contenant un attribut temporaire"""

    def __init__(self):
        self._attribut1 = "x"
        self._attribut2 = "y"
        self._attributtemp = 5

    # option 1:modifie le dictionnaire des attributs avant la sérialisation. Le dictionnaire des attributs enregistré
    # est celui que nous avons modifié avec la valeur de notre attribut temporaire à0.
    def __getstate__(self):
        """Renvoie le dico des attribut à serialiser"""
        dic_attribut = dict(self.__dict)
        dic_attribut['attributtemp'] = 0
        return dic_attribut

    # option 2:on modifie le dictionnaire d'attributs après la désérialisation
    def __setstate__(self, dict_attr):
        """Méthode appelée lors de la désérialisation de l'objet"""
        dict_attr["attribut_temporaire"] = 0
        self.__dict__ = dict_attr




# a completer avec la fonction pickel !!!

tempo = Temp()
dir(tempo)
__getstate__(tempo)


# L'héritage de classe


class Agent(Personne):
    """ Herite de la classe Personne"""
    def __init__(self, nom, prenom, matricule):
        Personne.__init__(self, nom, prenom)
        self.matricule = matricule
    def __str__(self): # si on ne definit cette methode, celle de la classe Personne est utilisée
        return('hello Agent {}'.format(self.matricule))

agent1 = Agent('bond', 'james', '007')
print(agent1)

issubclass(Personne,Agent) #FALSE
issubclass(Agent,Personne) # TRUE
isinstance(agent1, Personne)

class MaClasseHeritee(MaClasseMere1, MaClasseMere2): # cherche les methode dans la class 1 puis 2


# Creation de generateur


def intervalle(borne_inf, borne_sup):
    """Générateur parcourant la série des entiers entre borne_inf et borne_sup.
    Note: borne_inf doit être inférieure à borne_sup"""

    borne_inf += 1
    while borne_inf < borne_sup:
        yield borne_inf
        borne_inf += 1

def intervalle(inf,sup):
    """Generateur qui peut sauter des valeurs"""
    print('rentre dans le generator')
    inf += 1
    while inf <= sup:
        print('yield')
        valeur_rescue = (yield inf) # on peut envoyer une valeur a notre generateur par send
        if valeur_rescue is not None: # generator a recu une valeur
            inf = valeur_rescue
        else:
            inf += 1
            print ('incremente')

generator = intervalle(10,20)
for i in generator:
    if i == 15:
        generator.send(18) # on passe directement de 5 à 8
    print(i)



##### TP

class Dico():
    """ dico ordone"""

    def __init__(self):
        print('on passe par le constructeur')
        self.cles = []
        self.values = []

    def __str__(self):
        """affichage avec print"""
        return repr(self)

    def __repr__(self):
        """affichage sans print"""
        message = "{"
        premier=True
        for i in range(0,len(self.cles)):
            if not premier:
                message += ", "
            message += str(self.cles[i]) + ": "+ str( self.values[i])
            premier=False
        message += '}'
        return message

    def __delitem__(self, key):
        """delete the key and value associated"""
        if key not in self.cles:
            raise KeyError(' la cle {} nesxite pas!'.format(key))
        else:
            index = self.cles.index(key)
            del self.values[index]
            del self.cles[index]

    def __setitem__(self, key, value):
        """add an item, erase any existing key with same name"""
        if key in self.cles:
            index = self.cles.index(key)
            del self.values[index]
            del self.cles[index]
            # --> comment remplacer par del key???
        self.cles.append(key)
        self.values.append(value)

    def __getitem__(self, key):
        index = self.cles.index(key)
        return self.values[index]

    def __contains__(self, key):
        """check if the key is in the dico"""
        # if key in self.cles:
        #     return True
        # else:
        #     return False

        # plus simple:
        return key in self.cles

    def __len__(self):
        """Taille du dico"""
        return len(self.cles)

    def keys(self):
        L=[]
        for i in range(0,len(self.cles)):
            L.append(self.cles[i])
        return L
    keys = staticmethod(keys)

    def value(self):
        return list(self.values)

    def items(self):
        """Renvoie un générateur contenant les couples (cle, valeur)"""
        for i, cle in enumerate(self.cles):
            val = self.values[i]
        yield (cle, val)


    def __add__(self, other):
        if type(other) is not type(self):
            raise TypeError ("Le format du dico a ajouter nest pas conforme")

        self.cles.append(other.cles)
        self.values.append(other.values)




    def __iter__(self):
        """Méthode de parcours de l'objet. On renvoie l'itérateur des clés"""
        return iter(self.cles) # on renvoie l'iterateur des cles






fruits = Dico()
fruits
fruits['pomme'] = 52
fruits["poire"] = 34
fruits["prune"] = 128
fruits["melon"] = 15
print(fruits)
fruits
fruits['pomme']
del fruits["melon"]
'pomme' in fruits
len(fruits)

fruits.keys(fruits)
fruits.value(fruits)
for cle in fruits:
    print(cle)

fruits2 = Dico()
fruits2['fraises'] = 52
fruits + fruits2



################################### DECORATEUR ####################


# utilisation 1: Renvoie une fonction de substitution
def obsolete(fonction_origine):
    """Décorateur levant une exception pour noter que la fonction_origine
    est obsolète"""

    def fonction_modifiee():
        raise RuntimeError("la fonction {0} est obsolète !".format(fonction_origine))

    return fonction_modifiee


@obsolete
def salut():
    print('salut')

# Expression regulieres
import re
if re.search('^cha', 'chat') is not None:
    print('trouve!')

numero = ""

expression = "^0[0-9]([ .-]?[0-9]{2}){4}$"
while re.search(expression, numero) is None:
    numero = input('Numero de tel:')

re.sub('o','a','hello')
texte = """
nom='Task1', id=8
nom='Task2', id=31
nom='Task3', id=127"""

print(re.sub(r"id=", r"id[", texte))
print(re.sub(r"id=(?P<id>[0-9])", r"id=\g<id>", texte)) #(?P<nomdugroupe>) puis pour appler \g<nomdugroupe>
print(re.sub(r"id=(?P<id>[0-9]+)", r"id[\g<id>]", texte))
