import pytest
from jass.game.const import *
from jass.game.game_sim import GameSim
from jass.game.game_util import convert_one_hot_encoded_cards_to_int_encoded_list, deal_random_hand
from jass.game.rule_schieber import RuleSchieber

from bots.heuristic_bots import util


@pytest.fixture
def hand() -> np.ndarray:
    hand = np.zeros(36)
    for card in (D6, D10, DA, HJ, S8, SJ, SK, SA, C7):
        hand[card] = 1
    return hand


@pytest.fixture
def game() -> GameSim:
    np.random.seed(1)
    rule = RuleSchieber()
    game = GameSim(rule=rule)
    game.init_from_cards(hands=deal_random_hand(), dealer=NORTH)
    return game


def choose_trump(trump: int, game: GameSim):
    game.action_trump(trump)


def play_cards(game: GameSim, *, nr_of_cards: int = 4):
    for _ in range(nr_of_cards):
        valid_cards = game.rule.get_valid_cards_from_state(game.state)
        game.action_play_card(np.random.choice(np.flatnonzero(valid_cards)))


def test_better_card_same_color_first_higher_no_trump():
    card_1 = D10
    card_2 = D6
    trump = CLUBS
    card = util.get_better_card(card_1, card_2, trump)
    assert card == card_1


def test_better_card_same_color_first_lower_no_trump():
    card_1 = D6
    card_2 = D10
    trump = CLUBS
    card = util.get_better_card(card_1, card_2, trump)
    assert card == card_2


def test_better_card_same_color_first_9_second_jack_trump():
    card_1 = D9
    card_2 = DJ
    trump = DIAMONDS
    card = util.get_better_card(card_1, card_2, trump)
    assert card == card_2


def test_better_card_same_color_first_higher_une_ufe():
    card_1 = D10
    card_2 = D6
    trump = UNE_UFE
    card = util.get_better_card(card_1, card_2, trump)
    assert card == card_2


def test_better_card_different_color_first_lower_no_trump():
    card_1 = D6
    card_2 = HA
    trump = CLUBS
    card = util.get_better_card(card_1, card_2, trump)
    assert card == card_1


def test_better_card_different_color_first_lower_first_trump():
    card_1 = D6
    card_2 = HA
    trump = DIAMONDS
    card = util.get_better_card(card_1, card_2, trump)
    assert card == card_1


def test_better_card_different_color_first_higher_second_trump():
    card_1 = HA
    card_2 = D6
    trump = DIAMONDS
    card = util.get_better_card(card_1, card_2, trump)
    assert card == card_2


def test_have_buur(hand):
    assert util.have_puur(hand).tolist() == [0, 1, 1, 0]


def test_have_buur_with_four(hand):
    assert util.have_puur_with_four(hand).tolist() == [0, 0, 1, 0]


def test_get_trump_selection_score(hand):
    results = []
    cards = convert_one_hot_encoded_cards_to_int_encoded_list(hand)
    for trump in range(6):
        results.append(util.get_trump_selection_score(cards, trump))
    assert results == [46, 53, 67, 35, 62, 36]


def test_get_played_trump_cards(game):
    choose_trump(DIAMONDS, game)
    play_cards(game, nr_of_cards=16)
    obs = game.get_observation()
    assert util.get_played_trump_cards(obs.trump, obs.tricks) == [1, 6, 2, 0, 7, 4, 3]


def test_get_remaining_trump_cards(game):
    choose_trump(DIAMONDS, game)
    play_cards(game, nr_of_cards=16)
    obs = game.get_observation()
    assert util.get_remaining_trump_cards(obs.trump, obs.tricks) == [5, 8]


def test_get_bock_cards(game):
    choose_trump(DIAMONDS, game)
    play_cards(game, nr_of_cards=18)
    obs = game.get_observation()
    hand = convert_one_hot_encoded_cards_to_int_encoded_list(obs.hand)
    assert util.get_bock_cards(hand, obs.tricks) == [18, 20]


def test_get_trump_cards(game):
    choose_trump(DIAMONDS, game)
    play_cards(game, nr_of_cards=20)
    obs = game.get_observation()
    hand = convert_one_hot_encoded_cards_to_int_encoded_list(obs.hand)
    assert util.get_trump_cards(hand, obs.trump) == [5]


def test_get_points_in_trick(game):
    choose_trump(DIAMONDS, game)
    play_cards(game, nr_of_cards=15)
    obs = game.get_observation()
    assert util.get_points_in_trick(obs.trump, obs.current_trick) == 16
