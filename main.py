import numpy as np
import random
import math
from scipy.special import softmax
import matplotlib.pyplot as plt
import dirtyPlot

import threading


sem = threading.Semaphore()

gridSize = [10, 10]
A = ["haut", "bas", "gauche", "droite"]

agentPos = (0, 0)
agentStartPos = agentPos

rewardPos = (5, 5)
rewardStartPos = rewardPos

baseDGValue = 0

probFail = 0.8 #0.8 selon l'énoncé


"""
#move correspondant plus à ce qu'a fait leslie en retro-ingeneerant son code
def move(vchoice) :
  global agentPos

  targetPos = agentPos

  if vchoice == "haut":
    targetPos = (agentPos[0], agentPos[1] + 1)
  if vchoice == "bas":
    targetPos = (agentPos[0], agentPos[1] - 1)
  if vchoice == "gauche":
    targetPos = (agentPos[0] - 1, agentPos[1])
  if vchoice == "droite":
    targetPos = (agentPos[0] + 1, agentPos[1])
  if random.random() < probFail:
    r = random.random()
    if r < 0.25 :
      targetPos = (targetPos[0] + 1, targetPos[1])
    if r >= 0.25 and r < 0.5 :
      targetPos = (targetPos[0] - 1, targetPos[1])
    if r >= 0.5 and r < 0.75 :
      targetPos = (targetPos[0], targetPos[1] + 1)
    if r >= 0.75 :
      targetPos = (targetPos[0], targetPos[1] - 1)
  if targetPos[0] < gridSize[0] and targetPos[1] < gridSize[1] and targetPos[0] >= 0 and targetPos[1] >= 0:
    agentPos = targetPos
"""

def move(vchoice) :
  global agentPos

  targetPos = agentPos

  if vchoice == "haut":
    targetPos = (agentPos[0], agentPos[1] + 1)
  if vchoice == "bas":
    targetPos = (agentPos[0], agentPos[1] - 1)
  if vchoice == "gauche":
    targetPos = (agentPos[0] - 1, agentPos[1])
  if vchoice == "droite":
    targetPos = (agentPos[0] + 1, agentPos[1])

  if targetPos[0] < gridSize[0] and targetPos[1] < gridSize[1] and targetPos[0] >= 0 and targetPos[1] >= 0:
    if random.random() < probFail:
      # se trompe de case
      possibleFail = []

      if targetPos[0] + 1 < gridSize[0] and targetPos[1] < gridSize[1] and targetPos[0] + 1 >= 0 and targetPos[1] >= 0:
        possibleFail.append((targetPos[0] + 1, targetPos[1]))
      if targetPos[0] - 1 < gridSize[0] and targetPos[1] < gridSize[1] and targetPos[0] - 1 >= 0 and targetPos[1] >= 0:
        possibleFail.append((targetPos[0] - 1, targetPos[1]))
      if targetPos[0] < gridSize[0] and targetPos[1] + 1 < gridSize[1] and targetPos[0] >= 0 and targetPos[1] + 1 >= 0:
        possibleFail.append((targetPos[0], targetPos[1] + 1))
      if targetPos[0] < gridSize[0] and targetPos[1] - 1 < gridSize[1] and targetPos[0] >= 0 and targetPos[1] - 1 >= 0:
        possibleFail.append((targetPos[0], targetPos[1] - 1))
      targetPos = random.choice(possibleFail)
    agentPos = targetPos

#-----------------------------------------------------------------------------------------------------------
# Qlearning
#-----------------------------------------------------------------------------------------------------------

choice = -1
choice_tm1 = -1
rew = 0

S_tm1 = ''
#dictionnaire des couples états action et leurs valeurs
Qdict = {}
alpha = 0.9 #0.9  0.1 sur le Qlearning dans le code lisp
gamma = 0.9 #0.95 0.9 sur le Qlearning dans le code lisp
T = 0.1 #0.1

def getMaxFromQinQTable(Q) :
  qval = [Qdict.get((q, a), 0) for (q, a) in Qdict.keys() if Q == q]
  if qval == [] :
    return 0
  else :
    return np.max(qval)

def updateQ() :
  deltat = rew + gamma * getMaxFromQinQTable(agentPos)
  Qdict[(S_tm1, choice_tm1)] = (1-alpha) * Qdict.get((S_tm1, choice_tm1), 0) + alpha * deltat

"""def chooseQL() :
  sum = 0
  sumBoltz = np.sum([math.exp(Qdict.get((agentPos, a), 0)/T) for a in A])
  rand = random.random()
  for a in A :
    if rand < sum + math.exp(Qdict.get((agentPos, a), 0)/T)/sumBoltz :
      return a
    else :
      sum += math.exp(Qdict.get((agentPos, a), 0)/T)/sumBoltz

  print("erreur, devrait pas y avoir ce message")"""

def chooseQL() :
  x = [Qdict.get((agentPos, a), 0)/T for a in A]
  return random.choices(A, weights = softmax(x))[0]

#-----------------------------------------------------------------------------------------------------------
# DGlearning
#-----------------------------------------------------------------------------------------------------------

DGTable = {}

def initDGTable() :
  for a in A :
    for s1 in range(gridSize[0]) :
      for s2 in range(gridSize[1]) :
        DGTable[((s1, s2), a, (s1, s2))] = 0

def updateDG(UAGU = False) :
    if UAGU :
      for x in range(gridSize[0]):
        for y in range(gridSize[1]):
          anyGoal = (x,y)
          sem.acquire()
          DGTable[(S_tm1, choice_tm1, anyGoal)] = (1 - alpha) * DGTable.get((S_tm1, choice_tm1, anyGoal),
                                                                              baseDGValue) + alpha * (1 + DGTable.get(
            (agentPos, minDG(agentPos, anyGoal), anyGoal), baseDGValue))
          sem.release()
    else :
      sem.acquire()
      DGTable[(S_tm1, choice_tm1, rewardPos)] = (1 - alpha) * DGTable.get((S_tm1, choice_tm1, rewardPos),
                                                                          baseDGValue) + alpha * (1 + DGTable.get(
        (agentPos, minDG(agentPos, rewardPos), rewardPos), baseDGValue))
      sem.release()

def minDG(S, G) :
  #utiliser au sein d'un passage sous sémaphore !
  min = random.choice(A)
  for a in A :
    if DGTable.get((S, a, G), baseDGValue) < DGTable.get((S, min, G), baseDGValue) :
      min = a
  return min

"""def chooseDG() :
  sum = 0
  sem.acquire()
  max = np.max([DGTable.get((agentPos, a, rewardPos), baseDGValue) for a in A])
  sumBoltz = np.sum([math.exp((max - DGTable.get((agentPos, a, rewardPos), baseDGValue))/T) for a in A])
  rand = random.random()

  for a in A :
    if rand < sum + math.exp((max -DGTable.get((agentPos, a, rewardPos), baseDGValue))/T)/sumBoltz :
      sem.release()
      return a
    else :
      sum += math.exp((max-DGTable.get((agentPos, a, rewardPos), baseDGValue))/T)/sumBoltz
  print("erreur, devrait pas y avoir ce message")
  sem.release()"""

def chooseDG() :
  sem.acquire()
  x = [-DGTable.get((agentPos, a, rewardPos), baseDGValue)/T for a in A]
  sem.release()
  return random.choices(A, weights = softmax(x))[0]

def relaxation() :
  global DGTable
  for s1x in range(gridSize[0]) :
    for s1y in range(gridSize[1]) :
      s1 = (s1x, s1y)
      for s2x in range(gridSize[0]):
        for s2y in range(gridSize[1]):
          s2 = (s2x, s2y)
          for six in range(gridSize[0]):
            for siy in range(gridSize[1]):
              si = (six, siy)
              if s1 != s2 and s1!= si and si != s2 :
                for a in A :
                  sem.acquire()

                  temp = DGTable.get((s1, a, s2), baseDGValue)
                  """
                  DGTable[(s1, a, s2)] = min(temp,
                                                           DGTable.get((s1, a, si), baseDGValue)
                                                           + DGTable.get((si, minDG(si, s2), s2), baseDGValue)
                                                           )
                  """
                  if(DGTable.get((s1, a, si), baseDGValue) + DGTable.get((si, minDG(si, s2), s2), baseDGValue) < temp) :
                    #TODO : ça chie ici
                    #print("la relaxation trouve des trucs !")

                    #alors on dirait que sa formule de relaxation est pas très bien
                    #explication :
                    #DG(s1, a, s2) = min(DG(s1, a, s2), DG(s1, a, si) + min DG(si, a', s2))
                    #problème : ils sont initialisés à 0 par défaut, donc la relaxation est contreproductice, il faudrait les initiliser en fonction de leur distance ?
                    #c'est pas compliqué mais est-ce que c'est ce qu'elle a fait ?

                    #on est censé apprendre, je pense qu'il ne faut pas utiliser une heuristique pour initialiser la grille

                    #dans ce cas on ne devrait regarder que les valeurs présentes dans la table et ne pas initiliser

                    print(str(s1) + " " + str(si) + " " +str(s2))
                    print(str(DGTable.get((s1, a, si), baseDGValue) + DGTable.get((si, minDG(si, s2), s2), baseDGValue)) + "  <   " +  str(temp))

                    DGTable[(s1, a, s2)] = DGTable.get((s1, a, si), baseDGValue)+ DGTable.get((si, minDG(si, s2), s2),baseDGValue)


                  sem.release()

  #une fois fini, on recommence
  print("fin de relaxation")
  relaxation()



#-----------------------------------------------------------------------------------------------------------
# Main
#-----------------------------------------------------------------------------------------------------------

def main(method, nbRuns, nbticks, rewVal, agentBouge = False, rewardBouge = False, useAllGoalUpdate = False, useRelaxation= False, show = False):
  global agentPos
  global rewardPos
  global Qdict
  global DGTable
  global S_tm1
  global rew
  global choice
  global choice_tm1


  if show == True :
    dp = dirtyPlot.dirtyPlot(gridSize)

  # experiment related stuff
  run = 0
  ite = 0

  data = []
  meanData = []
  while run < nbRuns :
    runData = []
    run = run+1
    print("Run " + str(run))

    Qdict = {}

    sem.acquire()
    DGTable = {}
    initDGTable()
    sem.release()

    if(useRelaxation and method == "DG") :
      t = threading.Thread(target = relaxation)
      t.start()

    goaled = 0
    tick = 0


    while tick<nbticks:
      tick += 1
      ite += 1
      rew = 0

      if (agentPos == rewardPos):
        #print('***** REWARD REACHED *****')
        #print("in " + str(ite) + " iterations")
        goaled +=1
        runData.append(tick)
        meanData.append(tick)
        ite = -1
        rew = rewVal #random.random()

      if method == "Q" :
        if (ite != 0):
          updateQ()
        choice = chooseQL()
      if method == 'DG' :
        if (ite != 0):
          updateDG(UAGU= useAllGoalUpdate)
        choice = chooseDG()

      if show == True and tick >= 9000:
        dp.update(gridSize, S_tm1, rewardPos, choice_tm1, DGTable, Qdict, rew, method, 1)

      if (agentPos == rewardPos):
        if (agentBouge):
          agentPos = (random.randint(0, gridSize[0] - 1), random.randint(0, gridSize[1] - 1))
        else:
          agentPos = agentStartPos

        if (rewardBouge):
          rewardPos = (random.randint(0, gridSize[0] - 1), random.randint(0, gridSize[1] - 1))
        else:
          rewardPos = rewardStartPos
        S_tm1 = agentPos
      else :
        """print(choice)
        print(agentPos)
        for a in ["haut", "bas", "gauche", "droite"] :
          print(DGTable.get((agentPos, a, rewardPos ), 0))"""
        #choice = random.choice(["haut", "bas", "gauche", "droite"])

        S_tm1 = agentPos
        choice_tm1 = choice

        move(choice)

    data.append(runData)

    """
     if useRelaxation :
      print(testMT)
      #t.join() #bug ici, pourquoi je faisai ça déjà ?
    """

    print("number of goals attained : " + str(goaled))
    #print(np.mean(trialDuration))
  return meanData
#--------------------------------------

if __name__ == '__main__':

  nrun = 20
  nbucket = 100
  nticks = 10000
  rewVal = 1
  #random.seed(1234)
  print("--------------------------- Q -------------------------")
  #dataQ  = main("Q", nrun, nticks, rewVal, agentBouge=True, rewardBouge=False)
  dataQ  = main("Q", nrun, nticks, rewVal, agentBouge=True, rewardBouge=True)

  print("--------------------------- DG -------------------------")
  #dataDG = main("DG", nrun, nticks, rewVal, agentBouge=True, rewardBouge=False, useAllGoalUpdate=False, useRelaxation=False)
  dataDG = main("DG", nrun, nticks, rewVal, agentBouge=True, rewardBouge=True, useAllGoalUpdate=False, useRelaxation=True) #test de la relaxation

  """
  plt.hist([dataQ, dataDG], bins=100, histtype = 'step', label = ['Q', 'DG'])
  plt.legend()
  plt.show()
  #ça ne marche pas, je fais mon traitement moi même
  """

  finalQ = []
  finalDG = []

  for i in range(nbucket) :
    count = 0
    for e in dataQ :
      if e >= i*nticks/nbucket and e < (i+1)*nticks/nbucket :
        count += 1
    finalQ.append(count/(nrun*nticks/nbucket))
    count = 0
    for e in dataDG :
      if e >= i * nticks/nbucket and e < (i + 1) * nticks/nbucket:
        count += 1
    finalDG.append(count/(nrun*nticks/nbucket))

  print("mean Q : " + str(np.mean(finalQ)))
  print("mean DG : " + str(np.mean(finalDG)))

  plt.plot(finalQ, label = 'Q')
  plt.plot(finalDG, label = 'DG')
  plt.title('performances avec ' + str(nticks) + ' itérations \n gamma = ' + str(gamma) + ', rew = ' + str(rewVal) + ' et proba échec au mouvement = ' +str(probFail) )
  plt.ylabel('récompenses par tick')
  plt.xlabel('paquet de ' + str(nticks/nbucket) + ' ticks')
  plt.legend()
  plt.savefig('stdGam' + str(gamma) + 'Rew' + str(rewVal) + 'ProbFail' +str(probFail) +'.png')
  plt.show()


  #for a in A :
  #  print(DGTable.get(((5,5), a, (5,5)), 0))