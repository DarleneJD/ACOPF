# CÉLULA 1B — Aplicar patches de consistência Pyomo/OpenDSS
# Execute imediatamente DEPOIS da Célula 1, após carregar:
#   core, step1, step2

PATCH_FILE = CODE_DIR / "opf_runtime_patches.py"
patches = load_module_from_file("opf_runtime_patches", PATCH_FILE)
patches.apply(core, step2)

# Conferência rápida
print("core.build_opf patched:", getattr(core.build_opf, "_runtime_patched", False))
print("core.classify_inverters_from_opf patched:", getattr(core.classify_inverters_from_opf, "_runtime_patched", False))
print("core.export_pvsystems_to_dss patched:", getattr(core.export_pvsystems_to_dss, "_runtime_patched", False))
print("step2.case_policy_fixed_tap patched:", getattr(step2.case_policy_fixed_tap, "_runtime_patched", False))
print("step2.case_policy_optimal_tap patched:", getattr(step2.case_policy_optimal_tap, "_runtime_patched", False))
print("step2.export_validated_dss patched:", getattr(step2.export_validated_dss, "_runtime_patched", False))
