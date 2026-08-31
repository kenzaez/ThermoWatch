#Partie 1 — Python & Scripting (45 min)
# 1. Écris une fonction qui lit un fichier CSV, filtre les lignes où une colonne status vaut
# "active" , et écrit le résultat dans un nouveau CSV.
# 2. Explique la différence entre un script d'automatisation "scheduled" (cron) et un script
# "event-driven" (webhook). Donne un cas d'usage pour chacun dans un contexte
# immobilier/CRM.
# 3. Écris un script qui parcourt un dossier et renomme tous les fichiers .pdf selon le
# format YYYY-MM-DD_nomfichier.pdf à partir de leur date de création.
# 4. Qu'est-ce qu'un décorateur en Python ? Écris un décorateur @retry(n=3) qui réessaie
# une fonction jusqu'à 3 fois en cas d'exception.
# 5. Différence entre threading, multiprocessing, et asyncio en Python — dans quel cas
# utiliser lequel pour un outil d'automatisation qui appelle plusieurs APIs externes ?
#heres my solutions : 
import pandas as pd
def filter_csv(file):
    df = pd.read_csv(file)

#am i supposed to know this ?  un script d'automatisation "scheduled" (cron) et un script "event-driven" (webhook)