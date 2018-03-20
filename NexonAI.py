import random

class Player:
    DECK_SIZE = 5

    def init(self):
        self._deck = [ i for i in range( self.DECK_SIZE ) ]


    def turn(self):
        return self._deck.pop( random.randrange( len( self._deck ) ) )


class Game:
    ROUND_COUNT = 5
    GAME_COUNT = 10

    def __init__(self):
        self.winCount = 0
        self.players = [Player(), Player()]
        self.round = 0
        self.game = 0


    def roundStart(self):
        self.roundScore = 0

        self.round = 0

        for p in self.players:
            p.init()


    def gameFinish(self):
        print( "Result:{}".format( self.winCount ))


    def roundFinish(self):
        print("Round {}".format( self.round ) )
        if self.roundScore == 0:
            print( "Even" )
        elif self.roundScore > 0:
            self.winCount+=1
            print( "Win" )
        else:
            print( "Lose" )
        self.game+=1

        if self.game > self.GAME_COUNT:
            self.gameFinish()
            return True
        return False



    def nextRound(self):

        card0 = self.players[0].turn()
        card1 = self.players[1].turn()

        print("Round {},{}".format( card0, card1))

        if card0 > card1:
            self.roundScore+=1
        elif card0 < card1:
            self.roundScore-=1

        self.round+=1

        if self.round >= self.ROUND_COUNT:
            finish = self.roundFinish()
            self.roundStart()
            return finish

        return False




game = Game()
game.roundStart()

while( game.nextRound() == False ):
    pass





