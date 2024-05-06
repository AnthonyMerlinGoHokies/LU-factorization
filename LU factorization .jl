### A Pluto.jl notebook ###
# v0.19.36

using Markdown
using InteractiveUtils

# ╔═╡ 880b74ab-4706-493c-94c1-a9affaaea0a2
using LinearAlgebra

# ╔═╡ b5ed39bb-6300-4228-b6bc-0044aa871767
using MAT

# ╔═╡ 40808687-79f1-4b36-9373-a2626a288b76
#So our goal is solve the linear system Ax=b 
#by using Guassian elimnation (LU factorization) with partial pivoting

# ╔═╡ dad97288-bbe0-4ba9-8a9c-b3cb60594349
#and we need to make sure we implement Alogoritm 21.1 and solve the linear system

# ╔═╡ 415a94ab-a6df-439b-8199-4676b10a5e35
################################(Part1)#############################################

# ╔═╡ b0ca8ad5-6396-4218-955e-50b8264e3bea
#first let import out .mat file
file = matopen("/Users/anthonymerlin/Desktop/Numerical Analysis/hw7_beam.mat")


# ╔═╡ 8807c154-797a-4870-94aa-d026ee5c9b5f
beam_data = read(file)

# ╔═╡ 6c3abaaa-27b7-46cf-b4a2-e3f7b82f4d06
A = beam_data["A"]

# ╔═╡ d4a6c1de-88dc-47cc-a239-3e89f21fa204
b = beam_data["b"]

# ╔═╡ 50ef8d21-cd9b-4196-b584-101ee64ea186
real_x = A \ b

# ╔═╡ 7725bd4f-6253-49ff-bd3c-a5c5d25bb2ac
#okay now that we have it open let us start to implement algoritm 21.1 

# ╔═╡ 577e7d09-8ecf-4c22-af83-ff76ad0c4a4d
function GEPP(A::Matrix{Float64})
    # Get matrix dimensions
    m, n = size(A)
    
    U = copy(A)
    L = Matrix{Float64}(I, n, n)
    P = Matrix{Float64}(I, n, n)
    
    for k = 1:n-1
        # Find the row with the maximum absolute value in column k
        i_max = argmax(abs.(U[k:m, k])) + k - 1
        
        # Swap rows k and i_max in U
        U[k, :], U[i_max, :] = U[i_max, :], U[k, :]
        
        # Swap rows k and i_max in L and P
        L[k, 1:k-1], L[i_max, 1:k-1] = L[i_max, 1:k-1], L[k, 1:k-1]
        P[k, :], P[i_max, :] = P[i_max, :], P[k, :]
        
        # Perform elimination
        for i = k+1:m
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:n] -= L[i, k] * U[k, k:n]
        end
    end
    
    return U, L, P
end

# ╔═╡ 2a4468d2-c236-450b-bb05-ef84eb0a5e64
# can see that L is lower triangular and U is upper triangular
U, L, P = GEPP(A)  

# ╔═╡ ffacd4f5-275f-4f49-983d-be5bfeee856d
PA = P*A

# ╔═╡ 93074d7f-a383-4b9d-962e-9d4ee164aab7
LU = L*U

# ╔═╡ 75060aeb-720c-4c5d-8656-21a4cf33d158
norm(PA - LU)
#This PA and LU are nearly equal, and the LU  performed by the GEPP algorithm is accurate

# ╔═╡ 56c22fa7-730c-489f-8b27-ab6276f5cd62
#OK this is huge we just proved PA = LU Algoirthm 21.1 done :) 

# ╔═╡ b91b3b1d-c3ea-4630-ac27-1375740d25f0
###################################(Part 2)########################################

# ╔═╡ 3b73aa8c-bc12-4d1e-aa36-8763c88cc760
pb = P * b #make our permutated b here

# ╔═╡ c2cda996-63de-42be-9806-4a5392b3efbd
n = size(L, 1)

# ╔═╡ af1104a2-ebec-4818-80c8-94af69ffdfa5
function ComputeSum(x, i, L, n)
    sum = L[i,i] * x[i]
    for k in i+1:n
        sum += L[i,k] * x[k]
    end
    return sum
end

# ╔═╡ 434a7018-fd7f-45de-a162-fd54114da572
function ForwardSub(pb, L, n)
    x = zeros(size(pb))
    for i in 1:n
        x[i] = (pb[i] - ComputeSum(x, i, L, n)) / L[i,i]
    end
    return x
end

# ╔═╡ eb234d36-3e59-4864-9da4-0c7475726f45
y = ForwardSub(pb, L, n)

# ╔═╡ 4df6c98d-9d07-4755-a7d6-0faa02a6d3cc
#shows what we shoulve calculated L
y_backslash = L \ pb


# ╔═╡ e70eef5d-2aa2-4d0c-859e-4bc1b04add5a
# that we have our y lets use backwards subsitution to help us calculate our x from 
# Ux = y

# ╔═╡ ebda12eb-cdc2-491b-a317-d447843c998e
function BackwardSub(y, U, n)
    x = zeros(n)
    for i = n:-1:1
        x[i] = y[i]
        for j = i+1:n
            x[i] -= U[i,j] * x[j]
        end
        x[i] /= U[i,i]
    end
    return x
end

# ╔═╡ c23f2ce2-dcf0-4a51-822f-8e0b1b4cd695
x = BackwardSub(y, U, n)

# ╔═╡ 9a4911a5-9c37-4f06-ba0c-cf02557a2e24
ex = U \ y

# ╔═╡ dd4dba55-97b4-4381-95bc-bc501339b4b7
x2 = PA \ pb
#and here we are just checking what our explect x should look like

# ╔═╡ 005d2a0b-16c6-4aa0-918d-17ac012d1d62
#and lastly lets get out relative error 
re = norm(real_x - ex)/ norm(real_x)
# and based off this we can see our error is relatively high thus the error is due the stablity of the process of how we did things. so in conlusion use another method that may be more stable/accurate

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MAT = "23992714-dd62-5051-b70f-ba57cb901cac"

[compat]
MAT = "~0.10.6"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[BufferedStreams]]
git-tree-sha1 = "4ae47f9a4b1dc19897d3743ff13685925c5202ec"
uuid = "e1450e63-4bb3-523b-b2a4-4ffa8c0fd77d"
version = "1.2.1"

[[CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "59939d8a997469ee05c4b4944560a820f9ba0d73"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.4"

[[Compat]]
deps = ["Dates", "LinearAlgebra", "TOML", "UUIDs"]
git-tree-sha1 = "c955881e3c981181362ae4088b35995446298b80"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.14.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[HDF5]]
deps = ["Compat", "HDF5_jll", "Libdl", "MPIPreferences", "Mmap", "Preferences", "Printf", "Random", "Requires", "UUIDs"]
git-tree-sha1 = "e856eef26cf5bf2b0f95f8f4fc37553c72c8641c"
uuid = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
version = "0.17.2"

[[HDF5_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "LibCURL_jll", "Libdl", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "OpenSSL_jll", "TOML", "Zlib_jll", "libaec_jll"]
git-tree-sha1 = "82a471768b513dc39e471540fdadc84ff80ff997"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.14.3+3"

[[Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ca0f6bf568b4bfc807e7537f081c81e35ceca114"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.10.0+0"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MAT]]
deps = ["BufferedStreams", "CodecZlib", "HDF5", "SparseArrays"]
git-tree-sha1 = "ed1cf0a322d78cee07718bed5fd945e2218c35a1"
uuid = "23992714-dd62-5051-b70f-ba57cb901cac"
version = "0.10.6"

[[MPICH_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "656036b9ed6f942d35e536e249600bc31d0f9df8"
uuid = "7cb0a576-ebde-5e09-9194-50597f1243b4"
version = "4.2.0+0"

[[MPIPreferences]]
deps = ["Libdl", "Preferences"]
git-tree-sha1 = "8f6af051b9e8ec597fa09d8885ed79fd582f33c9"
uuid = "3da0fdf6-3ccc-4f1b-acd9-58baa6c99267"
version = "0.1.10"

[[MPItrampoline_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "77c3bd69fdb024d75af38713e883d0f249ce19c2"
uuid = "f1f71cc9-e9ae-5b93-9b94-4fe0e1ad3748"
version = "5.3.2+0"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[MicrosoftMPI_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f12a29c4400ba812841c6ace3f4efbb6dbb3ba01"
uuid = "9237b28f-5490-5468-be7b-bb81f5f5e6cf"
version = "10.1.4+2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OpenMPI_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "e25c1778a98e34219a00455d6e4384e017ea9762"
uuid = "fe0851c0-eecd-5654-98d4-656369965a5c"
version = "4.1.6+0"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3da7367955dcc5c54c1ba4d402ccdc09a1a3e046"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.13+1"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "71509f04d045ec714c4748c785a59045c3736349"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.7"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[libaec_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "46bf7be2917b59b761247be3f317ddf75e50e997"
uuid = "477f73a3-ac25-53e9-8cc3-50b2fa2566f0"
version = "1.1.2+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╠═40808687-79f1-4b36-9373-a2626a288b76
# ╠═dad97288-bbe0-4ba9-8a9c-b3cb60594349
# ╠═415a94ab-a6df-439b-8199-4676b10a5e35
# ╠═880b74ab-4706-493c-94c1-a9affaaea0a2
# ╠═b5ed39bb-6300-4228-b6bc-0044aa871767
# ╠═b0ca8ad5-6396-4218-955e-50b8264e3bea
# ╠═8807c154-797a-4870-94aa-d026ee5c9b5f
# ╠═6c3abaaa-27b7-46cf-b4a2-e3f7b82f4d06
# ╠═d4a6c1de-88dc-47cc-a239-3e89f21fa204
# ╠═50ef8d21-cd9b-4196-b584-101ee64ea186
# ╠═7725bd4f-6253-49ff-bd3c-a5c5d25bb2ac
# ╠═577e7d09-8ecf-4c22-af83-ff76ad0c4a4d
# ╠═2a4468d2-c236-450b-bb05-ef84eb0a5e64
# ╠═ffacd4f5-275f-4f49-983d-be5bfeee856d
# ╠═93074d7f-a383-4b9d-962e-9d4ee164aab7
# ╠═75060aeb-720c-4c5d-8656-21a4cf33d158
# ╠═56c22fa7-730c-489f-8b27-ab6276f5cd62
# ╠═b91b3b1d-c3ea-4630-ac27-1375740d25f0
# ╠═3b73aa8c-bc12-4d1e-aa36-8763c88cc760
# ╠═c2cda996-63de-42be-9806-4a5392b3efbd
# ╠═af1104a2-ebec-4818-80c8-94af69ffdfa5
# ╠═434a7018-fd7f-45de-a162-fd54114da572
# ╠═eb234d36-3e59-4864-9da4-0c7475726f45
# ╠═4df6c98d-9d07-4755-a7d6-0faa02a6d3cc
# ╠═e70eef5d-2aa2-4d0c-859e-4bc1b04add5a
# ╠═ebda12eb-cdc2-491b-a317-d447843c998e
# ╠═c23f2ce2-dcf0-4a51-822f-8e0b1b4cd695
# ╠═9a4911a5-9c37-4f06-ba0c-cf02557a2e24
# ╠═dd4dba55-97b4-4381-95bc-bc501339b4b7
# ╠═005d2a0b-16c6-4aa0-918d-17ac012d1d62
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
