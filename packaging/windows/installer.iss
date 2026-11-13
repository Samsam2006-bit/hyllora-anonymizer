; The anonymizer's Windows installer (Inno Setup 6). build.ps1 runs it on what packaging/stage.py and the bundled
; Python laid out in build\stage; it is not meant to be compiled by hand.
;
; Installs to C:\Program Files\Hyllora Anonymizer (python\, app\, models\, launcher.py), registers the service
; (service.xml) and starts it. The store's data is in C:\ProgramData\Hyllora\Anonymizer, readable only by SYSTEM and
; administrators (keys, streams, the empty store), and is left on uninstall.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef Stage
  #define Stage "build\stage"
#endif

[Setup]
AppId={{6E0C2B7A-3F4D-4E8B-9A51-0D2C7F1B8E43}
AppName=Hyllora anonymiserare
AppVersion={#AppVersion}
AppPublisher=Hyllora
DefaultDirName={autopf}\Hyllora Anonymizer
DisableDirPage=yes
DisableProgramGroupPage=yes
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\..\dist
OutputBaseFilename=HylloraAnonymizer-Setup-{#AppVersion}
Compression=lzma2/max
SolidCompression=yes
; torch for NVIDIA cards alone is several GB: past Inno's 2.1 GB per file it splits into the .exe and .bin files
; beside it, which must stay in the same folder
DiskSpanning=yes
DiskSliceSize=max
; the AGPL (anonymizer/LICENSE), shown before installing
LicenseFile=..\..\anonymizer\LICENSE
LZMANumBlockThreads=4
WizardStyle=modern
UninstallDisplayName=Hyllora anonymiserare

[Languages]
Name: "sv"; MessagesFile: "compiler:Languages\Swedish.isl"

[Files]
Source: "{#Stage}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "{#Stage}\..\WinSW-x64.exe"; DestDir: "{app}"; DestName: "hyllora-anonymizer-service.exe"; Flags: ignoreversion
Source: "service.xml"; DestDir: "{app}"; DestName: "hyllora-anonymizer-service.xml"; Flags: ignoreversion; AfterInstall: FillServiceXml

[Dirs]
Name: "{commonappdata}\Hyllora\Anonymizer"; Flags: uninsneveruninstall
Name: "{commonappdata}\Hyllora\Anonymizer\anonymizer"; Flags: uninsneveruninstall

[Icons]
Name: "{autoprograms}\Hyllora anonymiserare"; Filename: "http://127.0.0.1:8430/"

[Run]
; only SYSTEM (the service) and administrators may read the store's data: no inherited Users access
Filename: "{sys}\icacls.exe"; Parameters: """{commonappdata}\Hyllora\Anonymizer"" /inheritance:r /grant:r *S-1-5-18:(OI)(CI)F *S-1-5-32-544:(OI)(CI)F"; Flags: runhidden waituntilterminated
Filename: "{app}\hyllora-anonymizer-service.exe"; Parameters: "install"; Flags: runhidden waituntilterminated
Filename: "{app}\hyllora-anonymizer-service.exe"; Parameters: "start"; Flags: runhidden waituntilterminated
Filename: "http://127.0.0.1:8430/"; Description: "Öppna anonymiseraren"; Flags: postinstall shellexec nowait skipifsilent unchecked

[UninstallRun]
Filename: "{app}\hyllora-anonymizer-service.exe"; Parameters: "stop"; Flags: runhidden waituntilterminated; RunOnceId: "StopService"
Filename: "{app}\hyllora-anonymizer-service.exe"; Parameters: "uninstall"; Flags: runhidden waituntilterminated; RunOnceId: "RemoveService"

[UninstallDelete]
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\app"

[Code]
var
  SendPage: TInputQueryWizardPage;

procedure InitializeWizard;
var
  Old: AnsiString;
begin
  SendPage := CreateInputQueryPage(wpSelectTasks, 'Analysen',
    'Vart ska streckgubbarna skickas?',
    'Adressen till analysen och lådans nyckel (scripts/ingest_key.py på analysens dator). ' +
    'Lämna tomt för att bara anonymisera på den här datorn.');
  SendPage.Add('Analysens adress (till exempel https://analys.example.se):', False);
  SendPage.Add('Nyckel:', True);
  // a reinstall keeps the address it had (the key stays in its file when the field is left empty)
  if LoadStringFromFile(ExpandConstant('{commonappdata}\Hyllora\Anonymizer\anonymizer\ingest_url'), Old) then
    SendPage.Values[0] := Trim(String(Old));
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if (CurPageID = SendPage.ID) and (Trim(SendPage.Values[1]) <> '') and (Pos('.', SendPage.Values[1]) = 0) then
  begin
    MsgBox('Nyckeln ser inte ut som en lådnyckel (den har en punkt i sig).', mbError, MB_OK);
    Result := False;
  end;
end;

// after service.xml is copied ([Dirs] first, the service starts in [Run] after all files): the analysis' address
// into the service's environment, and the box's key into the data folder (kept on a reinstall if left empty)
procedure FillServiceXml;
var
  F, U, Url, Key: String;
  S: AnsiString;
begin
  Url := Trim(SendPage.Values[0]);
  SaveStringToFile(ExpandConstant('{commonappdata}\Hyllora\Anonymizer\anonymizer\ingest_url'), AnsiString(Url), False);
  StringChangeEx(Url, '&', '&amp;', True);
  StringChangeEx(Url, '<', '&lt;', True);
  StringChangeEx(Url, '"', '&quot;', True);
  F := ExpandConstant('{app}\hyllora-anonymizer-service.xml');
  LoadStringFromFile(F, S);
  U := String(S);
  StringChangeEx(U, '@INGEST_URL@', Url, True);
  SaveStringToFile(F, AnsiString(U), False);
  Key := Trim(SendPage.Values[1]);
  if Key <> '' then
    SaveStringToFile(ExpandConstant('{commonappdata}\Hyllora\Anonymizer\anonymizer\ingest_key'), AnsiString(Key), False);
end;

// a reinstall replaces a running service: stop and remove it before the files are copied
function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  Svc: String;
  Code: Integer;
begin
  Svc := ExpandConstant('{app}\hyllora-anonymizer-service.exe');
  if FileExists(Svc) then
  begin
    Exec(Svc, 'stop', '', SW_HIDE, ewWaitUntilTerminated, Code);
    Exec(Svc, 'uninstall', '', SW_HIDE, ewWaitUntilTerminated, Code);
  end;
  Result := '';
end;
