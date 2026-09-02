#include <stdio.h>
#include "l0.h"

int main() {
    // Создаем новое состояние виртуальной машины L0
    L0State *state = l0_new_state(); //[cite: 1]

    // Симуляция вызова внешней функции: считаем факториал числа 5 на стороне C
    int32_t factorial_result = 120;

    // Помещаем результат вычисления на вершину стека VM L0[cite: 1]
    l0_push_i32(state, factorial_result); //[cite: 1]

    // Извлекаем значение из стека по индексу 0 для проверки[cite: 1]
    int32_t out_value;
    if (l0_to_i32(state, 0, &out_value)) { //[cite: 1]
        printf("Передано в стек L0: %d\n", out_value);
    }

    // Безопасно освобождаем память VM[cite: 1]
    l0_free_state(state); //[cite: 1]
    return 0;
}
