import numpy as np
import random
import math
from scipy.special import softmax
import matplotlib.pyplot as plt
import dirtyPlot

import threading


sem = threading.BoundedSemaphore()

gridSize = [10, 10]
A = ["haut", "bas", "gauche", "droite"]

agentPos = (0, 0)
agentStartPos = agentPos

rewardPos = (5, 5)
rewardStartPos = rewardPos

baseDGValue = 0

probFail = 0.8 #0.8 selon l'énoncé

nb_relax = 100

def move(vchoice) :
  """
  bouge l'agent sur la grille suivant le choix
  :param vchoice: élément de A
  :return:
  """
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
#marche avec alpha=0.4 et gamma=0.995 étrangement
alpha = 0.9  #0.9 dans le papier
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
  """
  initialise la DG-table à 0 pour tout trio où l'état est l'objectif (nécessaire)
  :return:
  """
  for a in A :
    for s1 in range(gridSize[0]) :
      for s2 in range(gridSize[1]) :
        DGTable[((s1, s2), a, (s1, s2))] = 0

def updateDG(UAGU = False) :
    if UAGU :
      for x in range(gridSize[0]):
        for y in range(gridSize[1]):
          anyGoal = (x,y)
          DGTable[(S_tm1, choice_tm1, anyGoal)] = (1 - alpha) * DGTable.get((S_tm1, choice_tm1, anyGoal),
                                                                              baseDGValue) + alpha * (1 + DGTable.get(
            (agentPos, minDG(agentPos, anyGoal), anyGoal), baseDGValue))
    else :
      DGTable[(S_tm1, choice_tm1, rewardPos)] = (1 - alpha) * DGTable.get((S_tm1, choice_tm1, rewardPos),
                                                                          baseDGValue) + alpha * (1 + DGTable.get(
        (agentPos, minDG(agentPos, rewardPos), rewardPos), baseDGValue))

def minDG(S, G) :
  """
  :param S: etat courant
  :param G: objectif
  :return: l'action à faire qui a le plus de chances d'amener à S selon la DG-table
  """
  min = random.choice(A)
  for a in A :
    if DGTable.get((S, a, G), baseDGValue) < DGTable.get((S, min, G), baseDGValue) :
      min = a
  return min

def chooseDG() :
  x = [-DGTable.get((agentPos, a, rewardPos), baseDGValue)/T for a in A]
  return random.choices(A, weights = softmax(x))[0]

def relaxation() :
  """
  applique la relaxation sur tous les trio de valeurs de la DG table méthodiquement
  changer les sémaphores de place pour augmenter ou baisser la fréquence

  :return:
  """
  global DGTable
  for s1x in range(gridSize[0]) :
    for s1y in range(gridSize[1]) :
      s1 = (s1x, s1y)
      for s2x in range(gridSize[0]):
        for s2y in range(gridSize[1]):
          s2 = (s2x, s2y)
          #sem.acquire() #400
          for six in range(gridSize[0]):
            for siy in range(gridSize[1]):
              si = (six, siy)
              #sem.acquire() #4
              if s1 != s2 and s1!= si and si != s2 :
                for a in A :
                  sem.acquire() #1
                  temp = DGTable.get((s1, a, s2), baseDGValue)
                  mem1 = DGTable.get((s1, a, si), baseDGValue) + DGTable.get((si, minDG(si, s2), s2), baseDGValue)
                  if(mem1 < temp) :
                    #print("la relaxation trouve des trucs !")
                    #print(str(s1) + " " + str(si) + " " +str(s2))
                    #print(str(mem1) + "  <  " + str(temp))

                    DGTable[(s1, a, s2)] = mem1

                  t = threading.currentThread()
                  if(not getattr(t, "do_run", True)) :
                    sem.release()
                    return
                  sem.release() #1
              #sem.release() # 4
          #sem.release() # 400
  #une fois fini, on recommence
  print("fin de relaxation")
  relaxation()

def random_relax():
  """
  effectue une relaxation entre des états choisis aléatoirement
  :return:
  """
  global DGTable
  for i in range(nb_relax):
    s1x = random.randrange(gridSize[0])
    s2x = random.randrange(gridSize[0])
    s3x = random.randrange(gridSize[0])
    s1y = random.randrange(gridSize[1])
    s2y = random.randrange(gridSize[1])
    s3y = random.randrange(gridSize[1])
    s1 = (s1x, s1y)
    s2 = (s2x, s2y)
    s3 = (s3x, s3y)
    if s1 != s2 and s1 != s3 and s3 != s2:
      for a in A:
        temp = DGTable.get((s1, a, s2), baseDGValue)
        mem1 = DGTable.get((s1, a, s3), baseDGValue) + DGTable.get((s3, minDG(s3, s2), s2), baseDGValue)
        if (mem1 < temp):
          # print("la relaxation trouve des trucs !")
          # print(str(s1) + " " + str(si) + " " +str(s2))
          # print(str(mem1) + "  <  " + str(temp))

          DGTable[(s1, a, s2)] = mem1



#-----------------------------------------------------------------------------------------------------------
# Main
#-----------------------------------------------------------------------------------------------------------

def launch(method, nbRuns, nbticks, rewVal, agentBouge = False, rewardBouge = False, useAllGoalUpdate = False, relaxMode = 'no', show = False):
  """

  :param method: 'Q' ou 'DG', la méthode à employer
  :param nbRuns: entier, combien de fois les expériences devront être répétées (pour faire une moyenne)
  :param nbticks: entier, la durée de l'expérience
  :param rewVal: flottant, la valeur de la récompense pour le Q learning, 1 en général
  :param agentBouge: booléen, si la position de départ de l'agent change entre chaque objectif atteint
  :param rewardBouge: booléen, si la position de l'objectif change entre chaque objectif atteint
  :param useAllGoalUpdate: booléen si le DG learnign utilise le AllGoalUpdate
  :param relaxMode: 'fm' floyd warshal, 'rd' : random ou 'no' : aucun
  :param show: booléen : affiche la DG table ou la Q table sous forme de grille, utilisable à des fins de deboguage uniquement en objectif fixe
  :return: une liste contenant les ticks de chaque objectif atteint
  """
  global agentPos
  global rewardPos
  global Qdict
  global DGTable
  global S_tm1
  global rew
  global choice
  global choice_tm1


  if show == True and rewardBouge == False:
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

    DGTable = {}
    initDGTable()

    if(relaxMode == 'fm' and method == "DG") :
      sem.acquire()
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
          if relaxMode == "rd" :
            random_relax()
          if relaxMode == "fm" :
            sem.release()
            sem.acquire()
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

    if relaxMode == 'fm' :
      sem.release()
      t.do_run = False


    print("number of goals attained : " + str(goaled))
    #print(np.mean(trialDuration))
  return meanData
#--------------------------------------

if __name__ == '__main__':
  baseDGValue = 0
  nrun = 20
  nbucket = 100
  nticks = 10000
  rewVal = 1 #ne change jamais
  dyna = True
  alpha = 0.4 #0.9 0.4
  gamma = 0.995 #0.9 0.995
  AGA = True #True
  rel = 'rd' #'rd'

  #random.seed(1234)
  print("--------------------------- Q -------------------------")
  #dataQ  = launch("Q", nrun, nticks, rewVal, agentBouge=True, rewardBouge=False)
  dataQ  = launch("Q", nrun, nticks, rewVal, agentBouge=True, rewardBouge=dyna)

  print("--------------------------- DG -------------------------")
  #dataDG = launch("DG", nrun, nticks, rewVal, agentBouge=True, rewardBouge=False, useAllGoalUpdate=False, useRelaxation=False)
  dataDG = launch("DG", nrun, nticks, rewVal, agentBouge=True, rewardBouge=dyna, useAllGoalUpdate=AGA, relaxMode = rel) #test de la relaxation

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
  plt.figure(figsize=(6, 6))
  plt.plot(finalQ, label = 'Q')
  plt.plot(finalDG, label = 'DG')
  plt.title('performances avec ' + str(nticks) + ' itérations | dyna = ' + str(dyna) + '\n gamma = ' + str(gamma) + ' | alpha = ' + str(alpha) + ' | proba échec au mouvement = ' +str(probFail) + '\n allGoal : ' + str(AGA) + ' | relax : ' + rel + ' | baseDGval : ' + str(baseDGValue) )
  plt.ylabel('récompenses par tick')
  plt.xlabel('paquet de ' + str(nticks/nbucket) + ' ticks')
  plt.legend()

  plt.savefig('stdGam' + str(gamma) + 'al' + str(alpha) + 'dyna' + str(dyna) + 'ProbFail' +str(probFail) + 'AG' + str(AGA) + 'rel' + rel + 'BDGV' + str(baseDGValue)+'.png')
  plt.show()


  #for a in A :
  #  print(DGTable.get(((5,5), a, (5,5)), 0))