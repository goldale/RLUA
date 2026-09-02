# A collection of simple games written in Lua

## OXO - Noughts and Crosses

An implementation of noughts and crosses for the command line

`oxo.l0` is the L0 port. It replays a move sequence using zero-based cells and
prints the resulting X/O board and numeric winner (`1 = X`, `2 = O`).

## Connect four

An implementation of connect four. The computer opponent lacks some skills as it should be smarter than "Can't think of where to go. Just go anywhere". But at present I don't have the time to make things better

`connect_four.l0` is the L0 port. It is an interactive terminal game: human
plays `X`, computer plays `O`, and columns are numbered from `0` through `6`.
The opponent wins when possible, blocks an immediate human win, then prefers
the centre columns.

## Blocks

The same as 2048 or Threes. Slide the grid Up, Down, left and Right and get tiles with the same letter to merge and go up a letter. Game stops when you get to the letter 'z'

## Sudoku

A simple Sudoku solver, quite basic at the moment

`sudoku/sudoku.l0` is the first L0 port. It carries the original `easy0` puzzle
as a zero-indexed `vector<i8>` and calls the typed Sudoku standard-library
functions `sudoku_solve` and `sudoku_show`.
