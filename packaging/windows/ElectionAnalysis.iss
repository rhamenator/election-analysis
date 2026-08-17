#define AppName "Election Analysis MCP"
#ifndef AppVersion
  #define AppVersion "0.2.0"
#endif
[Setup]
UninstallDisplayIcon={app}\app-icon.ico
SetupIconFile=app-icon.ico
AppId={{39B58A62-1468-4875-B009-6402B4F76ED6}
AppName={#AppName}
AppVersion={#AppVersion}
DefaultDirName={autopf}\Election Analysis
DefaultGroupName={#AppName}
OutputDir=..\..\artifacts
OutputBaseFilename=ElectionAnalysis-{#AppVersion}-windows-x64-setup
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
[Files]
Source: "app-icon.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\dist\election-analysis-mcp.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\config.yaml"; DestDir: "{app}"; Flags: onlyifdoesntexist
[Icons]
Name: "{group}\Election Analysis MCP"; Filename: "{app}\election-analysis-mcp.exe"; IconFilename: "{app}\app-icon.ico"
