{
  description = "Micromamba-based Python environment with PyTorch (CPU)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        # Tools Nix should provide
        buildInputs = [
          pkgs.micromamba
          pkgs.bashInteractive
        ];

        # Optional: isolated mamba env directory inside repo
        MAMBA_ROOT_PREFIX = ".micromamba";

        shellHook = ''
          echo "Activating micromamba environment ‘pytorch-env’…"

          # Create env on first entry, if missing
          if ! micromamba env list | grep -q pytorch-env; then
            micromamba create -y -n pytorch-env python=3.11 \
              pytorch torchvision torchaudio cpuonly -c pytorch -c conda-forge
          fi

          # Activate it
          micromamba activate pytorch-env
        '';
      };
    };
}
