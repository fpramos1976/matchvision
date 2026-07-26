from app.models.court_type import CourtType
from app.models.match_configuration import MatchConfiguration


def get_match_configuration() -> MatchConfiguration:
    print("\n========== MatchVision ==========")
    print("Tipo de partida:")
    print("1 - Simples")
    print("2 - Duplas")

    while True:
        option = input("\nEscolha uma opção: ")

        if option == "1":
            return MatchConfiguration(court_type=CourtType.SINGLES)

        if option == "2":
            return MatchConfiguration(court_type=CourtType.DOUBLES)

        print("Opção inválida.")