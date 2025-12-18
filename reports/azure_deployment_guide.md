# 🚀 Guide Complet : Déployer ton App MLOps sur Azure (A à Z)

Ce guide est conçu pour t'accompagner étape par étape, même si tu n'as jamais utilisé Azure.

---

## 💎 Étape 1 : Activer Azure for Students (Gratuit)
1. Va sur [Azure for Students](https://azure.microsoft.com/free/students/).
2. Clique sur **Activate now**.
3. Connecte-toi avec ton adresse email académique (.edu ou .tn).
4. **Validation** : Tu recevras 100$ de crédit sans avoir à donner de carte bancaire.

---

## 🏗️ Étape 2 : Créer ta Machine Virtuelle (VM)
1. Dans la barre de recherche en haut, tape **Virtual Machines** et clique dessus.
2. Clique sur **Create** > **Azure virtual machine**.
3. **Configuration de base :**
    - **Resource Group** : Clique sur "Create new" et nomme-le `mlops-rg`.
    - **Virtual machine name** : `mlops-server`.
    - **Region** : `(Europe) West Europe` ou `North Europe`.
    - **Image** : `Ubuntu 22.04 LTS - x64 Gen2`.
    - **Size** : Choisis `Standard_B1s` (c'est la moins chère, ~8$/mois, parfaite pour ton crédit gratuit).
4. **Administrateur :**
    - **Authentication type** : `SSH public key`.
    - **Username** : `azureuser`.
    - **Key pair name** : `mlops-key`.
5. **Inbound Port Rules :**
    - Coche **SSH (22)**.
6. **IMPORTANT** : Clique sur **Review + create**, puis sur **Download private key and create resource**.
    - Tu vas télécharger un fichier `.pem`. Garde-le précieusement, c'est ta seule clé !

---

## 🔒 Étape 3 : Ouvrir les ports (Réseau)
Une fois la VM créée :
1. Va sur la page de ta VM dans le portail Azure.
2. Clique sur **Networking** (menu à gauche).
3. Clique sur **Add inbound port rule** :
    - **Destination port ranges** : `8000`
    - **Name** : `AllowFastAPI`
    - Clique sur **Add**.
4. Recommence pour le port `8501` :
    - **Destination port ranges** : `8501`
    - **Name** : `AllowStreamlit`
    - Clique sur **Add**.

---

## 💻 Étape 4 : Se connecter et Installer Docker
Ouvre un terminal (PowerShell ou Bash) sur ton PC là où tu as téléchargé la clé :

```bash
# Se connecter (remplace IP_VM par l'adresse IP publique affichée sur Azure)
ssh -i mlops-key.pem azureuser@IP_VM

# Une fois dans la VM, installe Docker :
sudo apt-get update
sudo apt install docker.io docker-compose -y
sudo usermod -aG docker $USER
exit
```
*Reconnecte-toi pour que les changements de groupe s'appliquent.*

---

## 🤖 Étape 5 : Configurer GitHub pour le déploiement
1. Ouvre ta clé `.pem` avec le bloc-notes et copie tout le texte.
2. Va sur ton repo GitHub > **Settings** > **Secrets and variables** > **Actions**.
3. Ajoute ces 3 secrets :
    - `VM_IP` : L'IP de ta VM.
    - `VM_USER` : `azureuser`.
    - `SSH_PRIVATE_KEY` : Colle le contenu du fichier `.pem`.

---

## 🔥 Étape 6 : Lancer le pipeline !
Il ne te reste plus qu'à pousser une mise à jour sur ton repo :
```bash
git add .
git commit -m "deploy: ready for azure"
git push origin master
```

**Ton app sera accessible ici :**
- Streamlit : `http://IP_VM:8501`
- API : `http://IP_VM:8000/docs`

---

### 💡 Conseil pour la présentation :
Mentionne que tu as choisi une **VM Linux** plutôt qu'un service managé pour avoir un contrôle total sur l'infrastructure Docker, ce qui est une pratique standard en **DevOps / MLOps**.
