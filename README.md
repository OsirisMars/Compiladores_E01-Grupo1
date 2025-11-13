# TrafficLang --- Compilador & Simulador (versão simplificada)

TrafficLang é um compilador/interpretador educativo para uma DSL de
controle (ex.: semáforos / robôs), com lexer, parser, verificação
semântica e um simulador simples.

------------------------------------------------------------------------

## Estrutura do repositório

    /
    ├── demo/                  # vídeo(s) demonstrativos (até 5 min)
    │   └── demo_short.mp4
    ├── docs/                  # documentação extra (opcional)
    ├── exemplos/              # exemplos .traffic (programas de teste)
    ├── src/                   # código-fonte principal (compilador / menu)
    ├── sensors/               # arquivos JSON de sensores (opcional)
    ├── tests/                 # (opcional) scripts de testes automatizados
    ├── README.md
    └── LICENSE

**Nota sobre `demo/`** A pasta `demo/` contém um vídeo curto (máximo
recomendado: 5 minutos) com demonstração do uso do compilador (menu,
execução de exemplos e análise). O arquivo dentro deve ter nome curto,
por exemplo `demo_short.mp4`. Se você publicar no GitHub e o repositório
for público, verifique o tamanho do vídeo --- GitHub recomenda arquivos
pequenos; se for maior, considere subir no YouTube/Vimeo e linkar aqui.

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

## Exemplo rápido

Arquivo `exemplo1.traffic` (colocar em `exemplos/`):

    SET x = 10;

    IF x > 5 AND x < 20 {
        MOVE;
    }

    STOP;

Sensores em `sensors/s1.json`:

``` json
{
  "temperatura": 30,
  "modo": "auto"
}
```

------------------------------------------------------------------------

## Demo (vídeo curto)

O vídeo demonstrativo está em `demo/demo_short.mp4` (até 5 minutos).\
Se preferir hospedar externamente, substitua pelo link do YouTube e
mencione aqui:

`Demo no YouTube: https://youtu.be/SEU_VIDEO`

------------------------------------------------------------------------

## Contribuindo

-   Abra issues para bugs / features
-   Faça forks e PRs (ex.: adicionar `ELSE`, persistência de variáveis,
    suporte a expressões aritméticas)
-   Mantenha testes em `tests/` e exemplos em `exemplos/`

------------------------------------------------------------------------

## Licença

MIT --- veja `LICENSE`.

------------------------------------------------------------------------

**Autor:** Gladiston Teles\
**Finalidade:** projeto educacional / demonstração de compiladores
(lexer, parser, semântica, interpretação)
