Lista1 = ["1P", "1C", "1O", "2P", "2C", "2O", "2E", "3P", "3C", "3O",
         "3E", "4P", "4C", "4O", "4E", "5P", "5C", "5O", "5E", "6P", "6C", "6O",
         "6E", "7P", "7C", "7O", "7E", "8P", "8C", "8O", "8E", "9P", "9C", "9O","9E","DP",
         "DC", "DO", "DE", "JP", "JC", "JO", "JE", "VP", "VC", "VO", "VE", "RP",   "RC", "RO", "RE",
         "1P", "1C", "1O", "1E", "2P", "2C", "2O", "2E", "3P", "3C", "3O", "3E",
         "4P", "4C", "4O", "4E", "5P", "5C", "5O", "5E", "6P", "6C", "6O", "6E", "7P", "7C",
         "7O", "7E", "8P", "8C", "8O", "8E", "9P", "9C", "9O", "9E", "DP", "DC", "DO", "DE",
         "JP", "JC", "JO", "JE", "VP", "VC", "VO", "VE", "RP", "RC", "RO", "RE", "1P", "1C",
         "1O", "1E", "2P", "2C", "2O", "2E", "3P", "3C", "3O", "3E", "4P", "4C", "4O", "4E",
         "5P", "5C", "5O", "5E", "6P", "6C", "6O", "6E", "7P", "7C", "7O", "7E", "8P", "8C",
         "8O", "8E", "9P", "9C", "9O", "9E", "DP", "DC", "DO", "DE", "JP", "JC", "JO", "JE",
         "VP", "VC", "VO", "VE", "RP", "RC", "RO", "RE"]

Lista2 = ['1C','1P','1O','1E','2C','2P','2O','2E','3C','3P','3O','3E','4C','4P','4O','4E'
           ,'5C','5P','5O','5E','6C','6P','6O','6E','7C','7P','7O','7E','8C','8P','8O','8E'
           ,'9C','9P','9O','9E''10C','10P','10O','10E''DC','DP','DO','DE','RC','RP','RO','RE'
           ,'VC','VP','VO','VE','AC','AP','AO','AE'
           ]

Lista3 = []
def baralho():
 for l1 in Lista1:
     Lista3.append(l1)
cont = 0
for l2 in Lista2:
    if l2 not  in Lista3:
     Lista3.append(l2)
    if Lista3 == Lista2:
            Lista3.sort()
            cont +=1
print("O número de Baralho Formado é: {}".format(cont))
print(Lista3)
baralho()
