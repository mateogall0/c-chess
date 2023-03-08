#include "board.h"

void boardDisplay()
{
  int i, j;

  puts("------------------");
  for (i = 0; i < 8; i++)
  {
    putchar('0' + (8 - i));
    for (j = 0; j < 8; j++)
    {
      putchar('|');
      switch(board[j][i]) {
        case 1: //pawn
        case -1:
          putchar(board[j][i] > 0 ? 'p' : 'P');
          break;
        case 2: //knight
        case -2:
          putchar(board[j][i] > 0 ? 'n' : 'N');
          break;
        case 3: //bishop
        case -3:
          putchar(board[j][i] > 0 ? 'b' : 'B');
          break;
        case 4: //rook
        case -4:
          putchar(board[j][i] > 0 ? 'r' : 'R');
          break;
        case 5:
        case -5:
          putchar(board[j][i] > 0 ? 'q' : 'Q');
          break;
        case 6:
        case -6:
          putchar(board[j][i] > 0 ? 'k' : 'K');
          break;
        default: //empty square
          putchar(' ');
          break;
      }
    }
    puts("|");
    puts("-|----------------");
  }
  puts("--a-b-c-d-e-f-g-h-");
}


int printBoard(char* move)
{
  if (!move) {
    boardDisplay();
    return 0;
  }
  return 0;
}