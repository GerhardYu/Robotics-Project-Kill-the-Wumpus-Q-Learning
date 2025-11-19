import random
import numpy as np
import cv2

size = 6
training_number = 50
alpha = 0.5 #learning rate
epsilon = 0.5 #probability of exploration
gamma = 0.7 #how much future rewards are valued
max_tries = 50
#movement 0 = up, 1 = down, 2 = left, 3 = right, 4 = shoot up, 5 = shoot down, 6 = shoot left, 7 = shoot right
actions = [0, 1, 2, 3, 4, 5, 6, 7]

class Player:
  current_location = []
  arrows = 10

class Wumpus:
  current_location = []
  shoot_score = 1000
  message = "You smell Wumpus"

class Hole:
  locations = []
  message = "You feel a draft of wind"

class Bats:
  locations = []
  message = "You hear the flapping of wings"

#generates map
def create_map(size):
  matrix_map = []
  for x in range(size):
    matrix_map.append([-1] * size)
  return matrix_map

#generates q table
def q_table_init(size):
  q_table = []
  matrix_size = size ** 2
  for x in range(matrix_size):
    q_table.append([0, 0, 0, 0, 0, 0, 0, 0])
  return q_table

#randomly generates a starting point for the player
def player_starting_point(matrix_map, size):
  starting_point = [random.randint(0, len(matrix_map) -1), random.randint(0, size-1)]
  matrix_map[starting_point[0]][starting_point[1]] = 0
  return starting_point

#randomly generates a starting point for wumpus
def wumpus_starting_point(matrix_map, size):
  wumpus_point = [random.randint(0, len(matrix_map) -1), random.randint(0, size-1)]
  while matrix_map[wumpus_point[0]][wumpus_point[1]] == 0:
    #makes sure wumpus is not generated on top of the player
    wumpus_point = [random.randint(0, len(matrix_map) -1), random.randint(0, size-1)]
  matrix_map[wumpus_point[0]][wumpus_point[1]] = -100
  return wumpus_point

#randomly generates a series of pit falls
def generate_holes(number, matrix_map, size):
  hole_locations = []
  for x in range(number):
    hole_point = [random.randint(0, len(matrix_map) - 1), random.randint(0, size - 1)]
     #will keep on generating hole if it generates to a tile already occupied
    while matrix_map[hole_point[0]][hole_point[1]] == 0 or matrix_map[hole_point[0]][hole_point[1]] == -100:
      hole_point = [random.randint(0, len(matrix_map) - 1), random.randint(0, size - 1)]
    matrix_map[hole_point[0]][hole_point[1]] = -200
    hole_locations.append(hole_point)

  return hole_locations

#generates bats at random locations
def generate_bats(number, matrix_map, size):
  bat_locations = []
  for x in range(number):
    bat_point = [random.randint(0, len(matrix_map) - 1), random.randint(0, size - 1)]
    #will keep on generating bats if it generates to a tile already occupied
    while matrix_map[bat_point[0]][bat_point[1]] == 0 or matrix_map[bat_point[0]][bat_point[1]] == -200 or matrix_map[bat_point[0]][bat_point[1]] == -100 or matrix_map[bat_point[0]][bat_point[1]] == -50:
      bat_point = [random.randint(0, len(matrix_map) - 1), random.randint(0, size - 1)]
    matrix_map[bat_point[0]][bat_point[1]] = -50
    bat_locations.append(bat_point)
  return bat_locations

#randomly moves player
def bat_move(player, matrix_map):
  random_point = [random.randint(0, size - 1), random.randint(0, size - 1)]
  #move to a random tile that is unoccupied
  while matrix_map[random_point[0]][random_point[1]] == -200 or matrix_map[random_point[0]][random_point[1]] == -50 or matrix_map[random_point[0]][random_point[1]] == -100:
    random_point = [random.randint(0, size - 1), random.randint(0, size - 1)]
  player.current_location = random_point
  return player.current_location

#moving logic
def action_take(matrix_map, q_table, current_location, action, size):
  if current_location[0] < size - 1 and current_location[1] == 0 and action == 2:
    #This shows that the current place of the user is in the left most part of the matrix
    return -200, current_location
  elif current_location[0] < size - 1 and current_location[1] == size - 1 and action == 3:
    #This shows that the current place of the user is in the right most part of the matrix
    return -200, current_location
  elif current_location[0] == 0 and current_location[1] < size - 1 and action == 0:
    #This shows that the current place of the user is in the upper most part of the matrix
    return -200, current_location
  elif current_location[0] == size - 1 and current_location[1] < size-1 and action == 1:
    #This shows that the current place of the user is in the lower most part of the matrix
    return -200, current_location
  elif current_location[0] == 0 and current_location[1] == 0 and (action == 0 or action == 2):
    #This shows that the current place of the user is in the upper left corner of the matrix
    return -200, current_location
  elif current_location[0] == 0 and current_location[1] == size - 1 and (action == 3 or action == 0):
    #This shows that the current place of the user is in upper right corner of the matrix
    return -200, current_location
  elif current_location[0] == size - 1 and current_location[1] == 0 and (action == 2 or action == 1):
    #This shows that the current place of the user is in lower left corner of the matrix
    return -200, current_location
  elif current_location[0] == size - 1 and current_location[1] == size - 1 and (action == 3 or action == 1):
    #This shows that the current place of the user is in lower right corner of the matrix
    return -200, current_location
  else:
    #This just returns the current user location
    if action == 0:
      new_location = [current_location[0] - 1, current_location[1]]
      return matrix_map[new_location[0]][new_location[1]], new_location
    elif action == 1:
      new_location = [current_location[0] + 1, current_location[1]]
      return matrix_map[new_location[0]][new_location[1]], new_location
    elif action == 2:
      new_location = [current_location[0], current_location[1] - 1]
      return matrix_map[new_location[0]][new_location[1]], new_location
    elif action == 3:
      new_location = [current_location[0], current_location[1] + 1]
      return matrix_map[new_location[0]][new_location[1]], new_location
    
def shoot_action(matrix_map, q_table, current_location, action, size, arrow_number, wumpus_location):
  #This function just shoots an arrow and checks if wumpus is there
  if action == 4:
    print("Shooting up")
    for x in range(current_location[0]):
      if matrix_map[x][current_location[1]] == -100:
        return True
  elif action == 5:
    print("Shooting down")
    for x in range(current_location[0], size):
      if matrix_map[x][current_location[1]] == -100:
        return True
  elif action == 6:
    print("Shooting left")
    for y in range(current_location[1]):
      if matrix_map[current_location[0]][y] == -100:
        return True
  elif action == 7:
    print("Shooting right")
    for y in range(current_location[1], size):
      if matrix_map[current_location[0]][y] == -100:
        return True
  return False

def check_if_near(matrix_map, wumpus, player):
  up = []
  down = []
  left = []
  right = []
  up.append([wumpus.current_location[0] - 1, wumpus.current_location[1]])
  down.append([wumpus.current_location[0] + 1, wumpus.current_location[1]])
  left.append([wumpus.current_location[0], wumpus.current_location[1] - 1])
  right.append([wumpus.current_location[0], wumpus.current_location[1] + 1])
  print("wumpus is near")

  if player.current_location[0] == up[0][0] and player.current_location[1] == up[0][1]:
    return True
  elif player.current_location[0] == down[0][0] and player.current_location[1] == down[0][1]:
    return True
  elif player.current_location[0] == left[0][0] and player.current_location[1] == left[0][1]:
    return True
  elif player.current_location[0] == right[0][0] and player.current_location[1] == right[0][1]:
    return True
  else:
    return False

#math logic for converting the map matrix into which row of the q table
def convert_matrix_to_q_table(current_location, q_table, size):
  q_table_location = current_location[0] * size + current_location[1]
  return q_table_location


def q_learning(player, wumpus, holes, bats, state_number = 8,size=size, alpha=alpha, epsilon=epsilon, gamma=gamma, training_number=training_number, max_tries=max_tries):
  rewards = []
  q_table = q_table_init(size)
  number = 1

  for x in range(training_number):
    near = False
    hit = False
    matrix_map = create_map(size)
    #each time a new run starts generate a new map, player, wumpus, hole, bats locations
    if number != 1:
      player.current_location = player_starting_point(matrix_map, size)
      player.arrows = 10
      wumpus.current_location = wumpus_starting_point(matrix_map, size)
      holes.locations = generate_holes(2, matrix_map, size)
      bats.locations = generate_bats(2, matrix_map, size)
    reward = 0
    q_table_location = convert_matrix_to_q_table(player.current_location, q_table, size)

    for y in range(max_tries):
      near = check_if_near(matrix_map, wumpus, player)
      if random.uniform(0, 1) < epsilon and player.arrows > 0:
        action_taken = random.randint(0, state_number - 1)
      elif random.uniform(0, 1) < epsilon and player.arrows < 0:
        #limit action when no arrows
        action_taken = random.randint(0, 3)
      elif random.uniform(0, 1) > epsilon and player.arrows < 0:
        #limit action when no arrows
        action_taken = np.argmax(q_table[q_table_location][0:4])
      else:
        action_taken = np.argmax(q_table[q_table_location])
      
      if near == True and random.uniform(0, 1) < epsilon:
        action_taken = random.randint(4, state_number - 1)
      elif near == True and random.uniform(0, 1) > epsilon:
        action_taken = np.argmax(q_table[q_table_location][4::])

      if action_taken < 4:
        new_reward, new_location = action_take(matrix_map, q_table, player.current_location, action_taken, size)
      elif action_taken > 3:
        #checks if you hit wumpus
        hit = shoot_action(matrix_map, q_table, player.current_location, action_taken, size, player.arrows, wumpus.current_location)
        if hit == True:
          player.arrows = player.arrows - 1
          new_reward = wumpus.shoot_score
          new_location = player.current_location
          print("you hit wumpus")
        else:
          player.arrows = player.arrows - 1
          new_reward = -500
          new_location = player.current_location
          print("you missed wumpus")
      reward = reward + new_reward
      epsilon = epsilon - 0.00000000005
      q_table_location = convert_matrix_to_q_table(player.current_location, q_table, size)
      q_table_location_new = convert_matrix_to_q_table(new_location, q_table, size)
      q_table[q_table_location][action_taken] = (1 - alpha) * q_table[q_table_location][action_taken] + alpha * (new_reward + gamma * max(q_table[q_table_location_new]))
      img = create_world(player.current_location, wumpus.current_location, holes.locations, bats.locations)
      cv2.imshow("Game", img)
      cv2.waitKey(500)
      if new_location == wumpus.current_location or new_location in holes.locations:
        #start a new run if you died
        print("you died")
        player.current_location = new_location
        break
      elif hit == True:
        #start a new run if you killed wumpus
        player.current_location = new_location
        break
      elif new_location in bats.locations:
        player.current_location = new_location
        player.current_location = bat_move(player, matrix_map)
      else:
        player.current_location = new_location
    print("Run Done")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    rewards.append(reward)
    number = number + 1
  return rewards, q_table

cell_size = 100

def create_world(player_location, wumpus_location, hole_location, bat_location):
  img = np.zeros((size*cell_size, size*cell_size, 3), dtype=np.uint8)

  for x in range(size):
    for y in range(size):
      top_left = (y* cell_size, x*cell_size)
      bottom_right = ((y+1)*cell_size, (x+1)*cell_size)

      color = (255, 255, 255)
      if [x,y] in hole_location:
        color = (0, 0, 255)
      if [x,y] == player_location:
        color = (255, 0, 0)
      if [x,y] == wumpus_location:
        color = (0, 255, 0)
      if [x,y] in bat_location:
        color = (100, 100, 100)

      cv2.rectangle(img, top_left, bottom_right, color, -1)
      cv2.rectangle(img, top_left, bottom_right, (0,0,0), 1)  # grid lines
  return img

def main():
  player = Player()
  wumpus = Wumpus()
  holes = Hole()
  bats = Bats()
  matrix = create_map(size)
  player.current_location = player_starting_point(matrix, size)
  wumpus.current_location = wumpus_starting_point(matrix, size)
  holes.locations = generate_holes(2, matrix, size)
  bats.locations = generate_bats(2, matrix, size)
  rewards, q_table =q_learning(player, wumpus, holes, bats)
  print("***************************")
  print("Rewards below")
  print(rewards)
  print("***************************")
  print("Q table below")
  print(q_table)
  



  
if __name__ == "__main__":
  main()