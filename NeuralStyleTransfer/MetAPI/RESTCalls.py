import requests
import random
import os


class RESTCalls:
    def GetRandomArtDocument(self):
        """ Method gets MET collection total, then loops
            randomly getting an art's json until we find
            one with an image and that's in public domain.

            Returns:
                Random art object
        """

        collection = requests.get(
            "https://collectionapi.metmuseum.org/public/collection/v1/search?medium=Paintings&q=*")
        collectionIds = collection.json()['objectIDs']

        hasValidImgUrl = False

        while(not hasValidImgUrl):
            try:
                randInt = random.randint(0, len(collectionIds))
                document = requests.get(
                    "https://collectionapi.metmuseum.org/public/collection/v1/objects/" + str(collectionIds[randInt]))
                documentJson = document.json()
                while(not documentJson['primaryImage']
                      or not documentJson['isPublicDomain']):
                    randInt = random.randint(0, len(collectionIds))
                    document = requests.get(
                        "https://collectionapi.metmuseum.org/public/collection/v1/objects/" + str(collectionIds[randInt]))
                    documentJson = document.json()
                hasValidImgUrl = True
            except:
                print("Retrying...")

        return document.json()
