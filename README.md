# TrafficLang --- Compilador & Simulador (versão simplificada)

TrafficLang é um compilador/interpretador educativo para uma DSL de
controle (ex.: semáforos / robôs), com lexer, parser, verificação
semântica e um simulador simples.

------------------------------------------------------------------------

## Estrutura do repositório

    /
    ├── demo/
    ├── docs/
    ├── src/
    ├── README.md

**Nota sobre `demo/`** A pasta `demo/` contém um vídeo curto (máximo
recomendado: 5 minutos) com demonstração do uso do compilador (menu,
execução de exemplos e análise).

------------------------------------------------------------------------

## Como executar (rápido)

1.  Clone:

``` bash
git clone https://github.com/SEU_USUARIO/TrafficLang.git
cd TrafficLang
```

2.  Execute:

``` bash
python src/compiladores_simples.py
```

3.  Use o menu:

-   `1` --- Carregar código (digitar / colar; termine com `END`)
-   `2` --- Carregar sensores (JSON)
-   `3` --- Mostrar tokens
-   `4` --- Mostrar AST
-   `5` --- Mostrar erros semânticos
-   `6` --- Executar simulador
-   `0` --- Sair


------------------------------------------------------------------------


**Autor:** Gladiston Teles\ Gabriel Moura \ Guilherme Chaves
**Finalidade:** projeto educacional / demonstração de compiladores
(lexer, parser, semântica, interpretação)
