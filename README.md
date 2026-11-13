# Hyllora anonymiserare (test)

Anonymiseraren ser kamerans bild och skickar bara siffror vidare: var personerna står och deras 17 leder (streckgubbar), plus en bild av den tomma butiken. Ingen bild av en person lämnar den.

## Starta på Windows (med NVIDIA-grafikkort)

1. **Code → Download ZIP** här ovanför, och packa upp mappen, till exempel på skrivbordet.
2. Dubbelklicka på **`start.bat`**. Första gången installeras allt som behövs (några GB, det tar en stund). Sedan går det fort.
3. Sidan öppnas i webbläsaren: **http://127.0.0.1:8430**. Fönstret med texten måste vara öppet så länge den körs.

Säger Windows "Windows skyddade din dator": *Mer information → Kör ändå* (filen kommer från internet).

Står det `Grafikkort: INGET` körs allt på processorn och blir långsamt. Uppdatera då NVIDIA-drivrutinen.

## Testa

- **En video:** *Ladda upp video*, välj den och tryck *Anonymisera*. Efteråt syns originalet och streckgubbarna sida vid sida.
- **Live:** under *Live-kameror* skriver du ett namn och kamerans adress (till exempel `rtsp://användare:lösen@192.168.1.64/stream`) och trycker *Starta*. Streckgubbarna syns live. Välj en uppladdad video i stället för *Egen adress* om du vill låtsas att den är en kamera.

Det som görs hamnar i `data/` i mappen och stannar på datorn.

## Ny version

**Stäng fönstret och dubbelklicka på `start.bat` igen.** Varje gång den startar letar den efter en ny version här och hämtar bara koden (några hundra KB). Det installerade, modellerna och `data/` behålls. Paketen installeras om bara när de har ändrats. Fönstret visar vilken version som körs.

## Licens

AGPL-3.0 (`LICENSE`), eftersom den kör Ultralytics YOLO. Kopierat från Hylloras huvudrepo med `packaging/testrepo.py`.
