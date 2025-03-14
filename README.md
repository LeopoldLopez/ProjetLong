# Réalisation et évaluation d’un prototype FaaS d’inférence  sur des modèles d’IA 

Lien vers tableau récap des sources : https://docs.google.com/document/d/1vFd0TpV6ojs8OmL6D97846g4U-1Am9knvjW2hn3FNAg/edit?pli=1&tab=t.0

# Pour lancer les scripts

Server_Folder/server_test est le serveur qui récupère les requêtes de Client_Folder/client_test.
Pour les tests, le client se lance comme ceci : Client_Folder/client_test sum 45
Cela veut dire que le client veut que le serveur fasse la somme(GPU) avec 45 paramètres (ici tout les paramètres seront égaux au chiffre 9, cela facilite le travail)

Server_Folder/server_test_inference est le serveur qui récupère les requêtes de Client_Folder/client
Le client envoie le script d'inférence, avec les paramètres, au serveur.

